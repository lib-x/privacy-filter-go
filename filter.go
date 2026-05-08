// Package privacyfilter 提供基于 openai/privacy-filter ONNX 模型的 PII 检测与脱敏能力。
//
// 快速上手：
//
//	f, err := privacyfilter.New(privacyfilter.Options{ModelDir: "./model"})
//	if err != nil { log.Fatal(err) }
//	defer f.Close()
//
//	result, err := f.Filter("My name is Alice, email: alice@example.com")
//	fmt.Println(result.Masked) // "My name is [PERSON], email: [EMAIL]"
package privacyfilter

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/lib-x/privacy-filter-go/internal/decoder"
	"github.com/lib-x/privacy-filter-go/internal/tokenizer"
)

// PrivacyFilter 是库的主入口，持有 tokenizer 和 ONNX 会话。
// 非并发安全；若需并发，请为每个 goroutine 创建独立实例。
type PrivacyFilter struct {
	opts    Options
	tok     *tokenizer.Tokenizer
	sess    *session
	allowed map[PIILabel]bool // nil 表示全部允许
}

// New 创建并初始化一个 PrivacyFilter。
// opts.ModelDir 必须包含 tokenizer.json 及所选 ONNX 文件。
func New(opts Options) (*PrivacyFilter, error) {
	// 应用默认值
	def := defaultOptions()
	if opts.Variant == "" {
		opts.Variant = def.Variant
	}
	if opts.ScoreThreshold == 0 {
		opts.ScoreThreshold = def.ScoreThreshold
	}
	if opts.ModelDir == "" {
		return nil, fmt.Errorf("privacyfilter: ModelDir must not be empty")
	}

	// 加载分词器
	tok, err := tokenizer.Load(opts.ModelDir)
	if err != nil {
		return nil, fmt.Errorf("privacyfilter: %w", err)
	}

	// 加载 ONNX 会话
	sess, err := loadSession(opts.ModelDir, opts.Variant, opts.OnnxRuntimeLib)
	if err != nil {
		tok.Close()
		return nil, fmt.Errorf("privacyfilter: %w", err)
	}

	// 构建标签白名单
	var allowed map[PIILabel]bool
	if len(opts.AllowedLabels) > 0 {
		allowed = make(map[PIILabel]bool, len(opts.AllowedLabels))
		for _, l := range opts.AllowedLabels {
			allowed[l] = true
		}
	}

	return &PrivacyFilter{
		opts:    opts,
		tok:     tok,
		sess:    sess,
		allowed: allowed,
	}, nil
}

// Close 释放底层资源，请在不再使用时调用。
func (f *PrivacyFilter) Close() {
	f.tok.Close()
	f.sess.destroy()
}

// Detect 对 text 执行 PII 检测，返回所有识别到的隐私片段。
func (f *PrivacyFilter) Detect(text string) ([]PIISpan, error) {
	result, err := f.process(text)
	if err != nil {
		return nil, err
	}
	return result.Spans, nil
}

// Filter 对 text 同时执行检测和脱敏，返回完整结果。
func (f *PrivacyFilter) Filter(text string) (*FilterResult, error) {
	return f.process(text)
}

// FilterBatch 批量处理多条文本（逐条串行，结果顺序与输入一致）。
func (f *PrivacyFilter) FilterBatch(texts []string) ([]*FilterResult, error) {
	results := make([]*FilterResult, len(texts))
	for i, t := range texts {
		r, err := f.process(t)
		if err != nil {
			return nil, fmt.Errorf("item %d: %w", i, err)
		}
		results[i] = r
	}
	return results, nil
}

// ---- 内部实现 ----

func (f *PrivacyFilter) process(text string) (*FilterResult, error) {
	if text == "" {
		return &FilterResult{Original: text, Masked: text}, nil
	}

	// 1. 分词
	enc, err := f.tok.Encode(text)
	if err != nil {
		return nil, fmt.Errorf("tokenize: %w", err)
	}

	// 2. ONNX 推理 → per-token softmax 概率
	probs, err := f.sess.run(enc.InputIDs, enc.AttentionMask)
	if err != nil {
		return nil, fmt.Errorf("inference: %w", err)
	}

	// 3. 构建 decoder.ClassInfo 表
	ciTable := buildDecoderClassTable()

	// 4. Viterbi 解码
	decoded := decoder.Decode(probs)

	// 5. 提取 token 级 span → 字节级 span
	rawSpans := decoder.ExtractSpans(decoded, ciTable, f.opts.ScoreThreshold)
	spans := f.toByteSpans(text, enc.Offsets, rawSpans)

	// 6. 过滤标签白名单
	if f.allowed != nil {
		filtered := spans[:0]
		for _, s := range spans {
			if f.allowed[s.Label] {
				filtered = append(filtered, s)
			}
		}
		spans = filtered
	}

	// 7. 生成脱敏文本
	masked := f.applyMask(text, spans)

	return &FilterResult{
		Original: text,
		Masked:   masked,
		Spans:    spans,
	}, nil
}

// toByteSpans 将 token 下标范围转换为原始字符串中的字节范围。
func (f *PrivacyFilter) toByteSpans(text string, offsets [][2]uint, raw []decoder.Span) []PIISpan {
	spans := make([]PIISpan, 0, len(raw))
	for _, r := range raw {
		startTok := r.StartTok
		endTok := r.EndTok

		if startTok >= len(offsets) || endTok >= len(offsets) {
			continue
		}

		byteStart := int(offsets[startTok][0])
		byteEnd := int(offsets[endTok][1])

		// 保证下标合法
		textLen := len(text)
		if byteStart < 0 || byteEnd > textLen || byteStart >= byteEnd {
			continue
		}

		spans = append(spans, PIISpan{
			Label: PIILabel(r.Label),
			Text:  text[byteStart:byteEnd],
			Start: byteStart,
			End:   byteEnd,
			Score: r.AvgScore,
		})
	}
	return spans
}

// applyMask 根据 MaskMode 将 spans 替换进原文，从后往前替换以保持偏移稳定。
func (f *PrivacyFilter) applyMask(text string, spans []PIISpan) string {
	if len(spans) == 0 {
		return text
	}

	// 将 text 转为 []byte 以便切片替换
	buf := []byte(text)

	// 从后往前处理，保持前面片段的字节偏移不变
	for i := len(spans) - 1; i >= 0; i-- {
		s := spans[i]
		replacement := f.replacement(s)
		buf = append(buf[:s.Start], append([]byte(replacement), buf[s.End:]...)...)
	}

	// 验证 UTF-8 合法性（避免切坏多字节字符时输出乱码）
	result := string(buf)
	if !utf8.ValidString(result) {
		// 退化：对每个 span 只做标签替换（字符串拼接方式）
		return f.applyMaskSafe(text, spans)
	}
	return result
}

func (f *PrivacyFilter) applyMaskSafe(text string, spans []PIISpan) string {
	var sb strings.Builder
	prev := 0
	for _, s := range spans {
		if s.Start > prev {
			sb.WriteString(text[prev:s.Start])
		}
		sb.WriteString(f.replacement(s))
		prev = s.End
	}
	sb.WriteString(text[prev:])
	return sb.String()
}

func (f *PrivacyFilter) replacement(s PIISpan) string {
	switch f.opts.Mode {
	case MaskRedact:
		return "***"
	case MaskRemove:
		return ""
	default: // MaskReplace
		return "[" + strings.ToUpper(shortLabel(s.Label)) + "]"
	}
}

// shortLabel 将 PIILabel 转换为简洁的占位符名称。
func shortLabel(l PIILabel) string {
	switch l {
	case LabelPerson:
		return "PERSON"
	case LabelEmail:
		return "EMAIL"
	case LabelPhone:
		return "PHONE"
	case LabelAddress:
		return "ADDRESS"
	case LabelURL:
		return "URL"
	case LabelDate:
		return "DATE"
	case LabelAccount:
		return "ACCOUNT"
	case LabelSecret:
		return "SECRET"
	default:
		return string(l)
	}
}

// buildDecoderClassTable 构建 decoder.ClassInfo 切片，与模型 id2label 对齐。
func buildDecoderClassTable() []decoder.ClassInfo {
	table := make([]decoder.ClassInfo, numClasses)
	table[0] = decoder.ClassInfo{Label: "", Tag: 0} // O
	labels := []string{
		"account_number", "private_address", "private_email",
		"private_person", "private_phone", "private_url",
		"private_date", "secret",
	}
	idx := 1
	for _, lbl := range labels {
		for tag := 1; tag <= 4; tag++ { // B I E S
			table[idx] = decoder.ClassInfo{Label: lbl, Tag: tag}
			idx++
		}
	}
	return table
}
