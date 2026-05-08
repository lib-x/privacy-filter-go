package privacyfilter

// PIILabel 是模型能识别的隐私信息类型。
type PIILabel string

const (
	LabelPerson  PIILabel = "private_person"
	LabelEmail   PIILabel = "private_email"
	LabelPhone   PIILabel = "private_phone"
	LabelAddress PIILabel = "private_address"
	LabelURL     PIILabel = "private_url"
	LabelDate    PIILabel = "private_date"
	LabelAccount PIILabel = "account_number"
	LabelSecret  PIILabel = "secret"
)

// MaskMode 控制检测到 PII 后如何处理原文。
type MaskMode int

const (
	// MaskReplace 用标签占位符替换，例如 [EMAIL]。
	MaskReplace MaskMode = iota
	// MaskRedact 用 *** 替换。
	MaskRedact
	// MaskRemove 直接删除该片段。
	MaskRemove
)

// ModelVariant 选择要加载的 ONNX 模型文件（精度/大小权衡）。
type ModelVariant string

const (
	// VariantFull     全精度 FP32，精度最高，约 1.82 GB
	VariantFull ModelVariant = "model.onnx"
	// VariantFP16     半精度，约 2.08 GB
	VariantFP16 ModelVariant = "model_fp16.onnx"
	// VariantQ4       4-bit 量化，约 917 MB，推荐用于生产
	VariantQ4 ModelVariant = "model_q4.onnx"
	// VariantQ4F16    4-bit 量化 + FP16，约 809 MB
	VariantQ4F16 ModelVariant = "model_q4f16.onnx"
	// VariantQuantized INT8 量化，约 1.62 GB
	VariantQuantized ModelVariant = "model_quantized.onnx"
)

// PIISpan 描述文本中一个检测到的隐私片段。
type PIISpan struct {
	Label PIILabel `json:"label"`
	Text  string   `json:"text"`
	Start int      `json:"start"` // 字节偏移（相对于原始字符串）
	End   int      `json:"end"`
	Score float32  `json:"score"` // 平均 token 置信度
}

// FilterResult 是对单条输入文本的处理结果。
type FilterResult struct {
	Original string    `json:"original"`
	Masked   string    `json:"masked"`   // 按 MaskMode 处理后的文本
	Spans    []PIISpan `json:"spans"`    // 检测到的所有 PII 片段
}

// Options 控制 PrivacyFilter 的行为。
type Options struct {
	// ModelDir 是存放 ONNX 模型文件和 tokenizer.json 的目录。
	ModelDir string

	// Variant 选择加载哪个 ONNX 文件，默认 VariantQ4。
	Variant ModelVariant

	// Mode 控制 Masked 字段中的脱敏方式，默认 MaskReplace。
	Mode MaskMode

	// AllowedLabels 若非空，则只检测指定类型的 PII；为空时检测全部 8 类。
	AllowedLabels []PIILabel

	// ScoreThreshold 低于此置信度的片段将被忽略，取值 0~1，默认 0.5。
	ScoreThreshold float32

	// OnnxRuntimeLib 指向 libonnxruntime 共享库路径；留空则自动搜索。
	OnnxRuntimeLib string
}

func defaultOptions() Options {
	return Options{
		Variant:        VariantQ4,
		Mode:           MaskReplace,
		ScoreThreshold: 0.5,
	}
}
