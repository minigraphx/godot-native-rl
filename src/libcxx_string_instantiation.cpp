// Make the zig-cross Linux build self-contained for std::string.
//
// libc++ marks the out-of-line std::basic_string<char> members (push_back / append / assign /
// resize / operator=) as `extern template`, with their definitions living in libc++.a. The zig
// cross-link for the Linux .so does not reliably pull that archive member in (host-dependent: a
// macOS-host build does; the Ubuntu-host CI build does not), so the .so links — undefined
// symbols are legal in a shared object — but fails to LOAD at runtime with
//   undefined symbol: std::__1::basic_string<...>::push_back
//
// An explicit instantiation definition forces those members to be emitted in this translation
// unit, resolving every reference at link time regardless of host. It is harmless on platforms
// whose C++ runtime already provides them (weak/COMDAT merge) — only the Linux zig build needs
// it, but emitting it everywhere keeps the build rule simple.
#include <string>

template class std::basic_string<char>;
