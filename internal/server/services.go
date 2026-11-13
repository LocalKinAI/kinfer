package server

import (
	"encoding/json"
	"log"
	"net"
	"net/http"
	"time"

	"github.com/LocalKinAI/kinfer/internal/services"
)

// The other programs on this machine, taken in by the front: see
// internal/services. Their state is in /api/ps and /api/services; adopting,
// releasing and stopping them is asked of /api/services/… from this machine
// only, as kinfer service does: what any host on the network could ask is to
// read.

// SetServices has the front manage the services in m: their room on the GPU
// is made with the runners', as one account.
func (s *Server) SetServices(m *services.Manager) {
	s.services = m
	m.Room = s.makeRoomForService
	m.Listen = OrderlyListener
}

// Services is the front's service manager, or nil.
func (s *Server) Services() *services.Manager { return s.services }

func (s *Server) handleServices(w http.ResponseWriter, r *http.Request) {
	if s.services == nil {
		writeJSON(w, http.StatusOK, map[string]any{"services": []any{}})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"services": s.services.Stats()})
}

// handleServiceAction answers POST /api/services/{adopt,release,stop}
// {"name": …, "idle": "30m"}.
func (s *Server) handleServiceAction(action string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		if !fromThisMachine(r) {
			writeJSON(w, http.StatusForbidden, map[string]string{"error": "services are adopted, released and stopped from this machine only"})
			return
		}
		if s.services == nil {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": services.ErrNoManager.Error()})
			return
		}
		var req struct {
			Name string `json:"name"`
			Idle string `json:"idle"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Name == "" {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "want {\"name\": …}"})
			return
		}
		var idle time.Duration
		if req.Idle != "" {
			d, err := time.ParseDuration(req.Idle)
			if err != nil {
				writeJSON(w, http.StatusBadRequest, map[string]string{"error": "idle: " + err.Error()})
				return
			}
			idle = d
		}
		var err error
		out := map[string]any{"name": req.Name}
		switch action {
		case "adopt":
			err = s.services.Adopt(req.Name, idle)
		case "release":
			var pid int
			pid, err = s.services.Release(req.Name)
			out["pid"] = pid
		case "stop":
			out["stopped"] = s.services.Stop(req.Name, "asked to")
		}
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, out)
	}
}

func fromThisMachine(r *http.Request) bool {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return false
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// makeRoomForService makes room on the GPU for a service about to start, as
// for a runner: the runner or the service unused longest goes first. A
// service with no measure yet starts in what there is, and is measured.
func (s *Server) makeRoomForService(name string, need uint64) {
	if need == 0 {
		return
	}
	s.makeRoom(name, need, "", name)
}

// makeRoom stops what is unused longest, runner or service, until need is
// free or nothing more can go; exceptPath and exceptService are the one
// starting.
func (s *Server) makeRoom(label string, need uint64, exceptPath, exceptService string) bool {
	for {
		free, total, ok := gpuMemory()
		if !ok {
			return true
		}
		if need > total {
			need = total
		}
		if free >= need {
			return true
		}
		r := s.runners.idlest(exceptPath)
		var svc string
		var since time.Time
		if s.services != nil {
			if n, t, ok := s.services.Idlest(); ok && n != exceptService {
				svc, since = n, t
			}
		}
		switch {
		case svc != "" && (r == nil || since.Before(r.used)):
			log.Printf("services: %s needs about %s of the GPU and %s is free: stopping %s, unused since %s",
				label, gb(need), gb(free), svc, since.Format("15:04:05"))
			if !s.services.StopUnused(svc, since, "room for "+label) {
				continue // used since it was found unused: look again
			}
			waitFreed(free)
		case r != nil:
			log.Printf("runner: %s needs about %s of the GPU and %s is free: stopping %s, unused since %s",
				label, gb(need), gb(free), modelName(r.path), r.used.Format("15:04:05"))
			s.runners.stopNow(r)
			waitFreed(free)
		default:
			return false
		}
	}
}
