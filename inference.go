package privacyfilter

import (
	"fmt"
	"math"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
)

// session 封装 ONNX Runtime 推理会话。
type session struct {
	sess     *ort.DynamicAdvancedSession
	seqLen   int64 // 当前批次的序列长度（动态）
}

// loadSession 从 modelDir 加载指定 variant 的 ONNX 模型。
func loadSession(modelDir string, variant ModelVariant, libPath string) (*session, error) {
	if libPath != "" {
		ort.SetSharedLibraryPath(libPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("onnxruntime init: %w", err)
	}

	modelPath := filepath.Join(modelDir, string(variant))

	// 输入：input_ids, attention_mask（都是 int64）
	// 输出：logits（float32）
	inputNames := []string{"input_ids", "attention_mask"}
	outputNames := []string{"logits"}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("session options: %w", err)
	}
	defer opts.Destroy()

	sess, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, opts)
	if err != nil {
		return nil, fmt.Errorf("create onnx session (%s): %w", modelPath, err)
	}
	return &session{sess: sess}, nil
}

// run 执行一次前向推理。
//
//   inputIDs, attnMask: 已分词的 int64 slice，长度均为 seqLen
//
// 返回 softmax 后的 per-token 概率矩阵，shape [seqLen][numClasses]。
func (s *session) run(inputIDs, attnMask []int64) ([][]float32, error) {
	seqLen := int64(len(inputIDs))

	shape := ort.NewShape(1, seqLen) // batch=1

	idsTensor, err := ort.NewTensor(shape, inputIDs)
	if err != nil {
		return nil, fmt.Errorf("create input_ids tensor: %w", err)
	}
	defer idsTensor.Destroy()

	maskTensor, err := ort.NewTensor(shape, attnMask)
	if err != nil {
		return nil, fmt.Errorf("create attention_mask tensor: %w", err)
	}
	defer maskTensor.Destroy()

	outShape := ort.NewShape(1, seqLen, numClasses)
	logitsTensor, err := ort.NewEmptyTensor[float32](outShape)
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer logitsTensor.Destroy()

	err = s.sess.Run(
		[]ort.Value{idsTensor, maskTensor},
		[]ort.Value{logitsTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("onnx run: %w", err)
	}

	raw := logitsTensor.GetData() // flat [1 * seqLen * numClasses]
	return reshapeAndSoftmax(raw, int(seqLen)), nil
}

// reshapeAndSoftmax 将平铺的 logit 数组转为 [seqLen][numClasses] 并做 softmax。
func reshapeAndSoftmax(flat []float32, seqLen int) [][]float32 {
	result := make([][]float32, seqLen)
	for t := 0; t < seqLen; t++ {
		row := flat[t*numClasses : (t+1)*numClasses]
		result[t] = softmax(row)
	}
	return result
}

func softmax(logits []float32) []float32 {
	maxV := logits[0]
	for _, v := range logits[1:] {
		if v > maxV {
			maxV = v
		}
	}
	out := make([]float32, len(logits))
	var sum float64
	for i, v := range logits {
		e := math.Exp(float64(v - maxV))
		out[i] = float32(e)
		sum += e
	}
	for i := range out {
		out[i] /= float32(sum)
	}
	return out
}

// destroy 释放会话资源。
func (s *session) destroy() {
	if s.sess != nil {
		s.sess.Destroy()
	}
}
