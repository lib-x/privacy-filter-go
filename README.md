# privacy-filter-go

基于 [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) ONNX 模型的 Go 语言 PII 检测与脱敏库。

## 能识别的隐私类型

| 标签 | 含义 |
|------|------|
| `private_person` | 自然人姓名 |
| `private_email` | 电子邮件地址 |
| `private_phone` | 电话号码 |
| `private_address` | 实体地址 |
| `private_url` | 个人 URL |
| `private_date` | 个人相关日期 |
| `account_number` | 账户 / 卡号 / IBAN |
| `secret` | API Key、密码、Token 等凭据 |

## 安装依赖

### 1. ONNX Runtime 共享库

```bash
# macOS (ARM)
brew install onnxruntime

# Ubuntu / Debian
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
tar xf onnxruntime-linux-x64-1.20.1.tgz
export ORT_LIB_PATH=$PWD/onnxruntime-linux-x64-1.20.1/lib
```

### 2. HuggingFace tokenizers 共享库（daulet/tokenizers 依赖）

```bash
# 参考 https://github.com/daulet/tokenizers#installation
# 库会自动从 GitHub Release 下载预编译的 libtokenizers.a
```

### 3. Go 依赖

```bash
go mod tidy
```

## 快速上手

```go
package main

import (
    "fmt"
    "log"

    pf "github.com/lib-x/privacy-filter-go"
)

func main() {
    // 首次运行自动下载模型（约 917 MB Q4 量化版）
    pf.DownloadModel("./model", pf.VariantQ4, func(f string, done, total int) {
        fmt.Printf("[%d/%d] %s\n", done, total, f)
    })

    filter, err := pf.New(pf.Options{
        ModelDir: "./model",
        Variant:  pf.VariantQ4,
        Mode:     pf.MaskReplace,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer filter.Close()

    result, _ := filter.Filter("Alice's email is alice@example.com, phone 555-0199.")
    fmt.Println(result.Masked)
    // → "[PERSON]'s email is [EMAIL], phone [PHONE]."

    for _, s := range result.Spans {
        fmt.Printf("%-18s %s\n", s.Label, s.Text)
    }
}
```

## API

### `pf.New(opts Options) (*PrivacyFilter, error)`

创建并初始化过滤器。

**Options 字段：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ModelDir` | `string` | 必填 | 模型文件所在目录 |
| `Variant` | `ModelVariant` | `VariantQ4` | ONNX 文件变体 |
| `Mode` | `MaskMode` | `MaskReplace` | 脱敏方式 |
| `AllowedLabels` | `[]PIILabel` | nil（全部） | 只检测指定类型 |
| `ScoreThreshold` | `float32` | `0.5` | 最低置信度阈值 |
| `OnnxRuntimeLib` | `string` | 自动搜索 | libonnxruntime 路径 |

### `filter.Filter(text string) (*FilterResult, error)`

检测 + 脱敏，返回完整结果。

### `filter.Detect(text string) ([]PIISpan, error)`

仅检测，返回 PII 片段列表。

### `filter.FilterBatch(texts []string) ([]*FilterResult, error)`

批量处理。

### `pf.DownloadModel(dir string, variant ModelVariant, progress func(...))`

从 HuggingFace 下载模型文件，已存在的文件自动跳过。

## 模型变体对比

| 变体 | 大小 | 精度 | 推荐场景 |
|------|------|------|----------|
| `VariantQ4` | ~917 MB | 略低 | **生产首选**（速度快，内存小）|
| `VariantQ4F16` | ~809 MB | 略低 | 内存受限场景 |
| `VariantQuantized` | ~1.62 GB | 中 | 精度与速度均衡 |
| `VariantFP16` | ~2.08 GB | 高 | GPU 加速场景 |
| `VariantFull` | ~1.82 GB | 最高 | 离线高精度分析 |

## 脱敏模式

| `MaskMode` | 效果示例 |
|------------|----------|
| `MaskReplace`（默认）| `alice@example.com` → `[EMAIL]` |
| `MaskRedact` | `alice@example.com` → `***` |
| `MaskRemove` | `alice@example.com` → `` |

## 注意事项

- 本库**不保证**合规性，仅作为多层隐私保护方案中的一层工具。
- 高敏感场景（医疗、金融、法律）建议额外人工审查。
- 模型主要针对英文优化，中文等非英文文本效果可能下降。
- 非并发安全，多 goroutine 请各自创建独立 `PrivacyFilter` 实例。

## License

Apache 2.0（与上游模型保持一致）
