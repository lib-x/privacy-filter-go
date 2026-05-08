package privacyfilter

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const hfBase = "https://huggingface.co/openai/privacy-filter/resolve/main"

// requiredFiles 按 variant 列出需要下载的文件。
var requiredFiles = map[ModelVariant][]string{
	VariantFull: {
		"model.onnx",
		"model.onnx_data",
		"model.onnx_data_1",
		"model.onnx_data_2",
	},
	VariantFP16: {
		"model_fp16.onnx",
		"model_fp16.onnx_data",
		"model_fp16.onnx_data_1",
	},
	VariantQ4: {
		"model_q4.onnx",
		"model_q4.onnx_data",
	},
	VariantQ4F16: {
		"model_q4f16.onnx",
		"model_q4f16.onnx_data",
	},
	VariantQuantized: {
		"model_quantized.onnx",
		"model_quantized.onnx_data",
	},
}

// commonFiles 不论哪个 variant 都需要的文件。
var commonFiles = []string{
	"tokenizer.json",
	"tokenizer_config.json",
	"special_tokens_map.json",
}

// DownloadModel 从 HuggingFace 下载模型文件到 destDir。
// 已存在的文件将被跳过。progress 为可选的进度回调（每个文件下载完后调用）。
func DownloadModel(destDir string, variant ModelVariant, progress func(file string, done, total int)) error {
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", destDir, err)
	}

	files := append(commonFiles, requiredFiles[variant]...)
	for i, name := range files {
		dest := filepath.Join(destDir, name)
		if _, err := os.Stat(dest); err == nil {
			// 文件已存在，跳过
			if progress != nil {
				progress(name, i+1, len(files))
			}
			continue
		}

		url := hfBase + "/onnx/" + name
		// tokenizer 相关文件在根目录
		if name == "tokenizer.json" || name == "tokenizer_config.json" || name == "special_tokens_map.json" {
			url = hfBase + "/" + name
		}

		if err := downloadFile(url, dest); err != nil {
			return fmt.Errorf("download %s: %w", name, err)
		}
		if progress != nil {
			progress(name, i+1, len(files))
		}
	}
	return nil
}

func downloadFile(url, dest string) error {
	resp, err := http.Get(url) //nolint:gosec
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d for %s", resp.StatusCode, url)
	}

	tmp := dest + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	defer func() {
		f.Close()
		os.Remove(tmp) // 若成功 rename 后这是个 no-op
	}()

	if _, err = io.Copy(f, resp.Body); err != nil {
		return err
	}
	f.Close()
	return os.Rename(tmp, dest)
}
