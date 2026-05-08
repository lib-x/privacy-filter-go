package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	pf "github.com/lib-x/privacy-filter-go"
)

func main() {
	modelDir := "./model"

	// ── 1. 首次运行：自动下载模型（~917 MB Q4 量化版）────────────────────────
	fmt.Println("Checking model files...")
	err := pf.DownloadModel(modelDir, pf.VariantQ4, func(file string, done, total int) {
		fmt.Printf("  [%d/%d] %s\n", done, total, file)
	})
	if err != nil {
		log.Fatalf("download model: %v", err)
	}

	// ── 2. 初始化过滤器 ───────────────────────────────────────────────────────
	filter, err := pf.New(pf.Options{
		ModelDir:       modelDir,
		Variant:        pf.VariantQ4,
		Mode:           pf.MaskReplace,  // [PERSON] [EMAIL] 等占位符
		ScoreThreshold: 0.6,
	})
	if err != nil {
		log.Fatalf("init filter: %v", err)
	}
	defer filter.Close()

	// ── 3. 单条文本检测 ───────────────────────────────────────────────────────
	text := "My name is Alice Wang, phone: +1-800-555-0199, email: alice@example.com. " +
		"She lives at 123 Maple St, Springfield. Her API key is sk-abc123xyz789."

	result, err := filter.Filter(text)
	if err != nil {
		log.Fatalf("filter: %v", err)
	}

	fmt.Println("\n── 原始文本 ──")
	fmt.Println(result.Original)

	fmt.Println("\n── 脱敏文本 ──")
	fmt.Println(result.Masked)

	fmt.Println("\n── 检测到的 PII 片段 ──")
	for _, s := range result.Spans {
		fmt.Printf("  %-18s | %-30s | score=%.3f | [%d, %d)\n",
			s.Label, s.Text, s.Score, s.Start, s.End)
	}

	// ── 4. 仅检测指定类型 ─────────────────────────────────────────────────────
	emailOnly, err := pf.New(pf.Options{
		ModelDir:      modelDir,
		Variant:       pf.VariantQ4,
		Mode:          pf.MaskRedact, // 替换为 ***
		AllowedLabels: []pf.PIILabel{pf.LabelEmail, pf.LabelSecret},
	})
	if err != nil {
		log.Fatalf("init email-only filter: %v", err)
	}
	defer emailOnly.Close()

	r2, _ := emailOnly.Filter(text)
	fmt.Println("\n── 仅脱敏 EMAIL + SECRET (MaskRedact) ──")
	fmt.Println(r2.Masked)

	// ── 5. 批量处理 ───────────────────────────────────────────────────────────
	batch := []string{
		"Contact John at john@corp.com or call 555-1234.",
		"Invoice #4892 for account 4111-1111-1111-1111.",
		"No personal info here, just a regular sentence.",
	}

	results, err := filter.FilterBatch(batch)
	if err != nil {
		log.Fatalf("batch filter: %v", err)
	}

	fmt.Println("\n── 批量处理结果 ──")
	for i, r := range results {
		fmt.Printf("[%d] %s\n    → %s\n", i, r.Original, r.Masked)
	}

	// ── 6. 仅获取 spans（不需要脱敏文本时更语义清晰）────────────────────────
	spans, err := filter.Detect("Send payment to IBAN DE89370400440532013000")
	if err != nil {
		log.Fatalf("detect: %v", err)
	}
	out, _ := json.MarshalIndent(spans, "", "  ")
	fmt.Println("\n── Detect() JSON 输出 ──")
	fmt.Println(string(out))

	os.Exit(0)
}
