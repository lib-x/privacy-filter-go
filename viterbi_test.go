package decoder_test

import (
	"testing"

	"github.com/lib-x/privacy-filter-go/internal/decoder"
)

// buildTable 构造与主包一致的 ClassInfo 表（33 项）。
func buildTable() []decoder.ClassInfo {
	table := make([]decoder.ClassInfo, 33)
	table[0] = decoder.ClassInfo{Label: "", Tag: 0}
	labels := []string{
		"account_number", "private_address", "private_email",
		"private_person", "private_phone", "private_url",
		"private_date", "secret",
	}
	idx := 1
	for _, lbl := range labels {
		for tag := 1; tag <= 4; tag++ {
			table[idx] = decoder.ClassInfo{Label: lbl, Tag: tag}
			idx++
		}
	}
	return table
}

// makeLogits 构造长度为 seqLen 的 logit 序列，
// 每个 token 在 classIdx 上分配高置信度，其余均分剩余。
func makeLogits(seqLen, classIdx int, highScore float32) [][]float32 {
	rows := make([][]float32, seqLen)
	for t := range rows {
		row := make([]float32, 33)
		low := (1.0 - highScore) / float32(32)
		for c := range row {
			row[c] = low
		}
		row[classIdx] = highScore
		rows[t] = row
	}
	return rows
}

// TestDecodeAllO 全部 token 应解码为 O 类。
func TestDecodeAllO(t *testing.T) {
	logits := makeLogits(5, 0, 0.99) // class 0 = O
	decoded := decoder.Decode(logits)
	for i, d := range decoded {
		if d.ClassIdx != 0 {
			t.Errorf("token %d: expected O(0), got %d", i, d.ClassIdx)
		}
	}
}

// TestDecodeSingleTokenSpan S-tag 应产生一个单 token span。
func TestDecodeSingleTokenSpan(t *testing.T) {
	// private_person S-tag = index 1 (B) + 3 (S offset) = 16
	// labelOrder: account(1-4), address(5-8), email(9-12), person(13-16)
	// S-person = 13 + 3 = 16
	sPersonIdx := 16

	// seqLen=3: O, S-person, O
	logits := [][]float32{
		makeLogits(1, 0, 0.99)[0],
		makeLogits(1, sPersonIdx, 0.99)[0],
		makeLogits(1, 0, 0.99)[0],
	}

	decoded := decoder.Decode(logits)
	table := buildTable()
	spans := decoder.ExtractSpans(decoded, table, 0.5)

	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	if spans[0].Label != "private_person" {
		t.Errorf("expected private_person, got %s", spans[0].Label)
	}
	if spans[0].StartTok != 1 || spans[0].EndTok != 1 {
		t.Errorf("expected tok [1,1], got [%d,%d]", spans[0].StartTok, spans[0].EndTok)
	}
}

// TestDecodeBIESpan B-I-E 序列应产生一个跨 3 token 的 span。
func TestDecodeBIESpan(t *testing.T) {
	// email: B=9, I=10, E=11, S=12
	bEmail, iEmail, eEmail := 9, 10, 11

	logits := [][]float32{
		makeLogits(1, 0, 0.99)[0],       // O
		makeLogits(1, bEmail, 0.99)[0],   // B-email
		makeLogits(1, iEmail, 0.99)[0],   // I-email
		makeLogits(1, eEmail, 0.99)[0],   // E-email
		makeLogits(1, 0, 0.99)[0],       // O
	}

	decoded := decoder.Decode(logits)
	table := buildTable()
	spans := decoder.ExtractSpans(decoded, table, 0.5)

	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	s := spans[0]
	if s.Label != "private_email" {
		t.Errorf("label: expected private_email, got %s", s.Label)
	}
	if s.StartTok != 1 || s.EndTok != 3 {
		t.Errorf("tok range: expected [1,3], got [%d,%d]", s.StartTok, s.EndTok)
	}
}

// TestScoreThreshold 低置信度 span 应被过滤。
func TestScoreThreshold(t *testing.T) {
	sPersonIdx := 16
	logits := [][]float32{
		makeLogits(1, sPersonIdx, 0.4)[0], // 低于 threshold=0.5
	}
	decoded := decoder.Decode(logits)
	table := buildTable()
	spans := decoder.ExtractSpans(decoded, table, 0.5)
	if len(spans) != 0 {
		t.Errorf("expected 0 spans (below threshold), got %d", len(spans))
	}
}
