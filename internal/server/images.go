package server

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/LocalKinAI/kinfer/internal/mlximage"
)

// ─── text to image ───────────────────────────────────────────────────────────
//
// POST /v1/images/generations, as OpenAI has it, answered by a model that
// paints (mlximage: Z-Image-Turbo on MLX). Behind the chat address the request
// goes to the picture model's runner, as speech goes to its own (runners.go):
// the model it names if that is one that paints, else this machine's default,
// fetched the first time it is asked for. The answer is PNG, base64, as
// gpt-image-1 answers; asked for a URL, it is a data: URL, there being no
// file to point at.

// paints is g's model as a painter, if it is one: the engine, not the
// wrapper that reloads it.
func paints(g generator) (*mlximage.Engine, bool) {
	if rt, ok := g.(*retrying); ok {
		g = rt.current()
	}
	e, ok := g.(*mlximage.Engine)
	return e, ok && e.Paints()
}

type imageRequest struct {
	Model          string  `json:"model"`
	Prompt         string  `json:"prompt"`
	N              int     `json:"n"`
	Size           string  `json:"size"`
	ResponseFormat string  `json:"response_format"`
	Seed           *uint64 `json:"seed"`  // kinfer's: the same seed, the same picture
	Steps          int     `json:"steps"` // kinfer's: denoising steps, 9 by default
}

func imageError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": map[string]string{"message": msg, "type": "invalid_request_error"}})
}

// parseSize reads OpenAI's "1024x1024"; "auto" and "" are 1024 a side.
func parseSize(s string) (w, h int, err error) {
	if s == "" || s == "auto" {
		return 1024, 1024, nil
	}
	a, b, ok := strings.Cut(strings.ToLower(s), "x")
	if ok {
		w, err1 := strconv.Atoi(strings.TrimSpace(a))
		h, err2 := strconv.Atoi(strings.TrimSpace(b))
		if err1 == nil && err2 == nil {
			return w, h, nil
		}
	}
	return 0, 0, fmt.Errorf("size %q: write width x height, as 1024x1024", s)
}

const maxImagesPerRequest = 4

func (s *Server) handleImages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if s.speech != SpeechOff { // answered in the picture model's runner
		if body, asked, ok := speechBody(w, r, 1<<20); ok {
			s.proxySpeech(w, r, kindImage, asked, body)
		}
		return
	}
	var req imageRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		imageError(w, http.StatusBadRequest, "send JSON with what to paint in \"prompt\" and the model in \"model\": "+err.Error())
		return
	}
	req.Model = s.runnerModel(r, req.Model)
	prompt := strings.TrimSpace(req.Prompt)
	switch {
	case req.Model == "":
		imageError(w, http.StatusBadRequest, "no model in \"model\"")
		return
	case prompt == "":
		imageError(w, http.StatusBadRequest, "nothing to paint in \"prompt\"")
		return
	}
	width, height, err := parseSize(req.Size)
	if err != nil {
		imageError(w, http.StatusBadRequest, err.Error())
		return
	}
	n := req.N
	if n == 0 {
		n = 1
	}
	if n < 1 || n > maxImagesPerRequest {
		imageError(w, http.StatusBadRequest, fmt.Sprintf("\"n\" is 1 to %d here: each picture takes seconds", maxImagesPerRequest))
		return
	}
	format := req.ResponseFormat
	switch format {
	case "", "b64_json", "url":
	default:
		imageError(w, http.StatusBadRequest, fmt.Sprintf("response_format %q: kinfer answers b64_json, or url as a data: URL", format))
		return
	}
	if req.Steps < 0 || req.Steps > 50 {
		imageError(w, http.StatusBadRequest, "\"steps\" is 1 to 50")
		return
	}

	eng, release, err := s.acquire(req.Model)
	if err != nil {
		writeJSON(w, acquireStatus(err), map[string]any{"error": map[string]string{"message": err.Error()}})
		return
	}
	defer release()
	p, ok := paints(eng)
	if !ok {
		imageError(w, http.StatusBadRequest, fmt.Sprintf("%s does not paint: a picture needs a text to image model, such as Z-Image-Turbo on a Mac", req.Model))
		return
	}
	seed := uint64(time.Now().UnixNano()) & (1<<53 - 1)
	if req.Seed != nil {
		seed = *req.Seed
	}
	var data []map[string]any
	start := time.Now()
	for i := 0; i < n; i++ {
		sd := seed + uint64(i)
		res, err := p.Paint(r.Context(), mlximage.Request{Prompt: prompt, Width: width, Height: height, Steps: req.Steps, Seed: &sd})
		if err != nil {
			if r.Context().Err() != nil {
				return
			}
			imageError(w, http.StatusBadRequest, err.Error())
			return
		}
		log.Printf("image: %d×%d, %d tokens, seed %d: %v (text %v, %d steps %v, decode %v)", res.Width, res.Height, res.Tokens, sd,
			res.Total.Round(time.Millisecond), res.Encode.Round(time.Millisecond), stepsOf(req.Steps),
			res.Denoise.Round(time.Millisecond), res.Decode.Round(time.Millisecond))
		b64 := base64.StdEncoding.EncodeToString(res.PNG)
		if format == "url" {
			data = append(data, map[string]any{"url": "data:image/png;base64," + b64, "revised_prompt": prompt})
		} else {
			data = append(data, map[string]any{"b64_json": b64, "revised_prompt": prompt})
		}
	}
	w.Header().Set("X-Kinfer-Seed", strconv.FormatUint(seed, 10))
	writeJSON(w, http.StatusOK, map[string]any{"created": start.Unix(), "data": data,
		"output_format": "png", "size": fmt.Sprintf("%dx%d", width, height)})
}

func stepsOf(n int) int {
	if n == 0 {
		return 9
	}
	return n
}
