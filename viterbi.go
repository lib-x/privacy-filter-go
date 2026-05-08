// Package decoder 实现基于 BIOES 标签序列的 Viterbi 约束解码，
// 将模型输出的 per-token logit 矩阵转换为连贯的隐私片段列表。
package decoder

import (
	"math"
)

// Span 是解码后的一个隐私片段（token 级别的下标）。
type Span struct {
	Label     string
	StartTok  int     // 包含，相对于 input_ids
	EndTok    int     // 包含
	AvgScore  float32 // 该 span 所有 token 置信度均值
}

// bioes tag 常量（与 labels.go 保持一致）
const (
	tagO = 0
	tagB = 1
	tagI = 2
	tagE = 3
	tagS = 4
)

// allowedNext[prev] 返回 prev 之后允许出现的 BIOES 状态集合。
// 规则：
//   O  → O, B, S
//   B  → I, E
//   I  → I, E
//   E  → O, B, S
//   S  → O, B, S
var allowedNext = [5][]int{
	tagO: {tagO, tagB, tagS},
	tagB: {tagI, tagE},
	tagI: {tagI, tagE},
	tagE: {tagO, tagB, tagS},
	tagS: {tagO, tagB, tagS},
}

// DecodedToken 是单个 token 最终选定的类别。
type DecodedToken struct {
	ClassIdx int
	Score    float32
}

// Decode 对整个序列执行约束 Viterbi 解码。
//
//   logits: shape [seqLen][numClasses]（已做 softmax 或原始 logit 均可，
//           内部取 log 后比较，结果不受单调变换影响）
//
// 返回每个 token 选定的类别下标及对应得分。
func Decode(logits [][]float32) []DecodedToken {
	seqLen := len(logits)
	numClasses := len(logits[0])

	// log 转换，避免浮点下溢
	logP := make([][]float64, seqLen)
	for t, row := range logits {
		logP[t] = make([]float64, numClasses)
		for c, v := range row {
			if v <= 0 {
				logP[t][c] = math.Log(1e-10)
			} else {
				logP[t][c] = float64(math.Log(float64(v)))
			}
		}
	}

	// dp[t][c] = 到 t 步选 c 类时的最大累积 log 概率
	dp := make([][]float64, seqLen)
	back := make([][]int, seqLen)
	for t := range dp {
		dp[t] = make([]float64, numClasses)
		back[t] = make([]int, numClasses)
		for c := range dp[t] {
			dp[t][c] = math.Inf(-1)
		}
	}

	// 初始化（t=0）
	for c := 0; c < numClasses; c++ {
		tag := tagOfClass(c)
		// 序列开头只允许 O, B, S
		if tag == tagO || tag == tagB || tag == tagS {
			dp[0][c] = logP[0][c]
		}
	}

	// 前向传播
	for t := 1; t < seqLen; t++ {
		for c := 0; c < numClasses; c++ {
			curTag := tagOfClass(c)
			curLabel := labelOfClass(c)
			for prevC := 0; prevC < numClasses; prevC++ {
				if dp[t-1][prevC] == math.Inf(-1) {
					continue
				}
				prevTag := tagOfClass(prevC)
				prevLabel := labelOfClass(prevC)

				if !transitionAllowed(prevTag, curTag, prevLabel, curLabel) {
					continue
				}
				score := dp[t-1][prevC] + logP[t][c]
				if score > dp[t][c] {
					dp[t][c] = score
					back[t][c] = prevC
				}
			}
		}
	}

	// 回溯
	best := make([]int, seqLen)
	best[seqLen-1] = argmaxFloat64(dp[seqLen-1])
	for t := seqLen - 2; t >= 0; t-- {
		best[t] = back[t+1][best[t+1]]
	}

	result := make([]DecodedToken, seqLen)
	for t, c := range best {
		result[t] = DecodedToken{ClassIdx: c, Score: logits[t][c]}
	}
	return result
}

// ExtractSpans 将解码后的 token 类别序列转换为隐私片段列表。
//
//   decoded:    Decode() 的返回值
//   classTable: 下标 → (label, bioes_tag) 的映射表
//   threshold:  最低平均置信度，低于此值的片段被丢弃
func ExtractSpans(decoded []DecodedToken, classTable []ClassInfo, threshold float32) []Span {
	var spans []Span
	inSpan := false
	var cur Span
	var scoreSum float32
	var scoreCount int

	flush := func() {
		if inSpan {
			cur.AvgScore = scoreSum / float32(scoreCount)
			if cur.AvgScore >= threshold {
				spans = append(spans, cur)
			}
			inSpan = false
			scoreSum = 0
			scoreCount = 0
		}
	}

	for t, dt := range decoded {
		ci := classTable[dt.ClassIdx]
		tag := ci.Tag
		label := ci.Label

		switch tag {
		case tagO:
			flush()
		case tagS:
			flush()
			spans = append(spans, Span{
				Label:    label,
				StartTok: t,
				EndTok:   t,
				AvgScore: dt.Score,
			})
		case tagB:
			flush()
			inSpan = true
			cur = Span{Label: label, StartTok: t, EndTok: t}
			scoreSum = dt.Score
			scoreCount = 1
		case tagI:
			if inSpan {
				cur.EndTok = t
				scoreSum += dt.Score
				scoreCount++
			}
		case tagE:
			if inSpan {
				cur.EndTok = t
				scoreSum += dt.Score
				scoreCount++
				flush()
			}
		}
	}
	flush()
	return spans
}

// ClassInfo 是 labels.go 中 classInfo 在 decoder 包内的镜像（避免循环依赖）。
type ClassInfo struct {
	Label string
	Tag   int // 0=O 1=B 2=I 3=E 4=S
}

// ---- 内部工具函数 ----

// tagOfClass 从类别下标提取 BIOES tag。
func tagOfClass(c int) int {
	if c == 0 {
		return tagO
	}
	return (c-1)%4 + 1 // B=1 I=2 E=3 S=4
}

// labelOfClass 从类别下标提取 label 名称（O 类返回空串）。
func labelOfClass(c int) string {
	if c == 0 {
		return ""
	}
	labels := []string{
		"account_number", "private_address", "private_email",
		"private_person", "private_phone", "private_url",
		"private_date", "secret",
	}
	return labels[(c-1)/4]
}

// transitionAllowed 检查 (prevTag, prevLabel) → (curTag, curLabel) 是否合法。
// 对于 I/E，还需保证 label 一致（不允许跨类别 Inside/End）。
func transitionAllowed(prevTag, curTag int, prevLabel, curLabel string) bool {
	allowed := allowedNext[prevTag]
	ok := false
	for _, a := range allowed {
		if a == curTag {
			ok = true
			break
		}
	}
	if !ok {
		return false
	}
	// I 或 E 时，label 必须与前驱相同
	if curTag == tagI || curTag == tagE {
		return prevLabel == curLabel
	}
	return true
}

func argmaxFloat64(s []float64) int {
	best, bestVal := 0, s[0]
	for i, v := range s[1:] {
		if v > bestVal {
			bestVal = v
			best = i + 1
		}
	}
	return best
}
