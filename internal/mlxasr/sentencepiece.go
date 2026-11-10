package mlxasr

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

// pieces reads the vocabulary of a SentencePiece model file (a ModelProto),
// in id order: what a CTC model's ids decode to. Only the pieces are read;
// nothing here encodes.
func readPieces(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []string
	for len(b) > 0 {
		field, wire, n := tag(b)
		if n <= 0 {
			return nil, fmt.Errorf("%s: not a SentencePiece model", path)
		}
		b = b[n:]
		switch wire {
		case 0: // varint
			_, m := binary.Uvarint(b)
			if m <= 0 {
				return nil, fmt.Errorf("%s: a broken varint", path)
			}
			b = b[m:]
		case 1:
			b = b[8:]
		case 5:
			b = b[4:]
		case 2: // length-delimited
			l, m := binary.Uvarint(b)
			if m <= 0 || int(l) > len(b)-m {
				return nil, fmt.Errorf("%s: a field longer than the file", path)
			}
			body := b[m : m+int(l)]
			b = b[m+int(l):]
			if field == 1 { // ModelProto.pieces
				out = append(out, pieceText(body))
			}
		default:
			return nil, fmt.Errorf("%s: wire type %d", path, wire)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s has no pieces", path)
	}
	return out, nil
}

func tag(b []byte) (field, wire int, n int) {
	v, n := binary.Uvarint(b)
	return int(v >> 3), int(v & 7), n
}

// pieceText is SentencePiece.piece (field 1) of a piece message.
func pieceText(b []byte) string {
	for len(b) > 0 {
		field, wire, n := tag(b)
		if n <= 0 {
			return ""
		}
		b = b[n:]
		switch wire {
		case 2:
			l, m := binary.Uvarint(b)
			if m <= 0 || int(l) > len(b)-m {
				return ""
			}
			if field == 1 {
				return string(b[m : m+int(l)])
			}
			b = b[m+int(l):]
		case 0:
			_, m := binary.Uvarint(b)
			b = b[m:]
		case 5:
			_ = math.Float32frombits(binary.LittleEndian.Uint32(b))
			b = b[4:]
		case 1:
			b = b[8:]
		default:
			return ""
		}
	}
	return ""
}

// decodePieces is SentencePiece's decoding: pieces joined, ▁ a space, byte
// pieces (<0xE4>) their bytes.
func decodePieces(vocab []string, ids []int32) string {
	var b []byte
	for _, id := range ids {
		if id < 0 || int(id) >= len(vocab) {
			continue
		}
		p := vocab[id]
		if len(p) == 6 && strings.HasPrefix(p, "<0x") && p[5] == '>' {
			if v, err := strconv.ParseUint(p[3:5], 16, 8); err == nil {
				b = append(b, byte(v))
				continue
			}
		}
		b = append(b, strings.ReplaceAll(p, "▁", " ")...)
	}
	return strings.TrimSpace(strings.ToValidUTF8(string(b), "�"))
}
