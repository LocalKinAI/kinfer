// Package dl opens native libraries the way each system does: dlopen where
// there is one, LoadLibraryEx on Windows. purego calls into a library on every
// system it supports but offers Dlopen and Dlsym only where those exist.
package dl
