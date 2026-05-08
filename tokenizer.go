// Package tokenizer 封装 HuggingFace tokenizers（via daulet/tokenizers CGO 绑定）。
package tokenizer

import (
	"fmt"
	"path/filepath"

	hftok "github.com/daulet/tokenizers"
)

// Tokenizer 封装了分词器，提供编码与 offset 映射。
type Tokenizer struct {
	tk *hftok.Tokenizer
}

// Encoding 是一次编码的结果。
type Encoding struct {
	InputIDs      []int64
	AttentionMask []int64
	// Offsets[i] 是第 i 个 token 在原始字符串中的字节范围 [start, end)。
	Offsets [][2]uint
}

// Load 从 modelDir 目录加载 tokenizer.json。
func Load(modelDir string) (*Tokenizer, error) {
	path := filepath.Join(modelDir, "tokenizer.json")
	tk, err := hftok.FromFile(path)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer from %s: %w", path, err)
	}
	return &Tokenizer{tk: tk}, nil
}

// Encode 对单条文本进行编码，返回 input_ids、attention_mask 和字节偏移。
func (t *Tokenizer) Encode(text string) (*Encoding, error) {
	enc, err := t.tk.EncodeWithOptions(
		text,
		true, // addSpecialTokens
		hftok.WithReturnAllAttributes(),
	)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}

	ids := enc.IDs
	inputIDs := make([]int64, len(ids))
	attnMask := make([]int64, len(ids))
	offsets := make([][2]uint, len(ids))

	rawOffsets := enc.Offsets
	for i, id := range ids {
		inputIDs[i] = int64(id)
		attnMask[i] = 1
		if i < len(rawOffsets) {
			offsets[i] = [2]uint{uint(rawOffsets[i][0]), uint(rawOffsets[i][1])}
		}
	}

	return &Encoding{
		InputIDs:      inputIDs,
		AttentionMask: attnMask,
		Offsets:       offsets,
	}, nil
}

// Close 释放底层资源。
func (t *Tokenizer) Close() {
	t.tk.Close()
}
