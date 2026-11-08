// SmoothMyVideo's update helper: the app copies the NEW version's copy of this exe to %TEMP% and runs it as it exits. A
// GUI-subsystem process, so it opens no console and holds nothing inside the app folder. Arguments: the app's pid, its
// folder, the unpacked new version's folder relative to it (update\<version>\x\<name>), 1 to start the new version
// afterwards, the exe's name. It waits for the app to exit, gives any process still running an exe from the app folder
// (the app's own hosts) ten seconds and then ends it, then replaces ONLY the files the old version shipped, as
// listed in its resources\release-files.txt: each listed file moves into update\<version>\old\, the new version's
// listed files move in (its own list included), the exe and app.asar last; everything else in the folder (the user's
// videos and renders, the one-click downloads, the model cache) is never touched. Every move is written to
// update\<version>\journal.txt first: a failed move rolls every earlier one back, and a journal left by a helper that
// stopped part-way is rolled back before a new attempt. A file still in use gets SMV_UPDATE_TRIES tries two seconds
// apart (default 30). A failure is noted in <folder>\update\failed.txt. Log: %TEMP%\smv-update.log. Exit 0 = swapped,
// 1 = not, 2 = bad arguments.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <shellapi.h>
#include <tlhelp32.h>

#include <algorithm>
#include <cwchar>
#include <cwctype>
#include <string>
#include <unordered_set>
#include <vector>

namespace
{
std::wstring g_log;
int g_tries = 30;

const wchar_t* const kList = L"resources\\release-files.txt";
const wchar_t* const kAsar = L"resources\\app.asar";

std::wstring join(const std::wstring& a, const std::wstring& b)
{
    return a + L"\\" + b;
}

std::wstring lower(std::wstring s)
{
    for (auto& c : s)
        c = (wchar_t)std::towlower(c);
    return s;
}

std::string utf8Of(const std::wstring& s)
{
    const int n = WideCharToMultiByte(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0, nullptr, nullptr);
    std::string out((size_t)(n > 0 ? n : 0), '\0');
    if (n > 0)
        WideCharToMultiByte(CP_UTF8, 0, s.c_str(), (int)s.size(), out.data(), n, nullptr, nullptr);
    return out;
}

std::wstring wideOf(const std::string& s)
{
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring out((size_t)(n > 0 ? n : 0), L'\0');
    if (n > 0)
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), out.data(), n);
    return out;
}

void writeAll(HANDLE h, const std::string& bytes)
{
    DWORD written = 0;
    WriteFile(h, bytes.data(), (DWORD)bytes.size(), &written, nullptr);
}

void logLine(const std::wstring& m)
{
    SYSTEMTIME t;
    GetLocalTime(&t);
    wchar_t stamp[32];
    swprintf_s(stamp, L"[%02d:%02d:%02d.%03d] ", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
    HANDLE h = CreateFileW(g_log.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE)
        return;
    writeAll(h, utf8Of(stamp + m + L"\r\n"));
    CloseHandle(h);
}

std::wstring lastError()
{
    return L"error " + std::to_wstring(GetLastError());
}

bool isFile(const std::wstring& p)
{
    const DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

bool readText(const std::wstring& p, std::string& out)
{
    HANDLE h =
        CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE)
        return false;
    LARGE_INTEGER size{};
    bool ok = GetFileSizeEx(h, &size) && size.QuadPart < (1 << 26);
    if (ok)
    {
        out.assign((size_t)size.QuadPart, '\0');
        DWORD got = 0;
        ok = out.empty() || (ReadFile(h, out.data(), (DWORD)out.size(), &got, nullptr) && got == out.size());
    }
    CloseHandle(h);
    return ok;
}

std::vector<std::string> lines(const std::string& text)
{
    std::vector<std::string> out;
    size_t at = 0;
    while (at < text.size())
    {
        size_t end = text.find('\n', at);
        if (end == std::string::npos)
            end = text.size();
        std::string line = text.substr(at, end - at);
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (!line.empty())
            out.push_back(line);
        at = end + 1;
    }
    return out;
}

// a version's release-files.txt: "<sha256> <size> <path relative to the app folder>" a line, '#' lines skipped; the
// list itself is part of the release too
bool readList(const std::wstring& file, std::vector<std::wstring>& out)
{
    std::string text;
    if (!readText(file, text))
        return false;
    for (const auto& line : lines(text))
    {
        if (line[0] == '#')
            continue;
        const size_t a = line.find(' ');
        const size_t b = a == std::string::npos ? a : line.find(' ', a + 1);
        if (b == std::string::npos || b + 1 >= line.size())
            return false;
        std::wstring rel = wideOf(line.substr(b + 1));
        std::replace(rel.begin(), rel.end(), L'/', L'\\');
        out.push_back(rel);
    }
    if (out.empty())
        return false;
    out.push_back(kList);
    return true;
}

void ensureParent(const std::wstring& path)
{
    for (size_t i = path.find(L'\\', 3); i != std::wstring::npos; i = path.find(L'\\', i + 1))
        CreateDirectoryW(path.substr(0, i).c_str(), nullptr);
}

// one rename with the tries; false = it never happened
bool moveFile(const std::wstring& from, const std::wstring& to)
{
    ensureParent(to);
    for (int n = 1;; n++)
    {
        if (MoveFileExW(from.c_str(), to.c_str(), 0))
            return true;
        const DWORD e = GetLastError();
        if (n >= g_tries || e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND || e == ERROR_ALREADY_EXISTS)
        {
            logLine(L"move \"" + from + L"\" -> \"" + to + L"\": error " + std::to_wstring(e));
            return false;
        }
        Sleep(2000);
    }
}

struct Swap
{
    std::wstring top, tree, old, journal;

    void note(wchar_t kind, const std::wstring& rel) const
    {
        HANDLE h = CreateFileW(journal.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ, nullptr, OPEN_ALWAYS,
                               FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH, nullptr);
        if (h == INVALID_HANDLE_VALUE)
            return;
        writeAll(h, utf8Of(std::wstring(1, kind) + L"\t" + rel + L"\r\n"));
        CloseHandle(h);
    }
    // O = an old release file moved out of the app folder, N = a new file moved in
    bool out(const std::wstring& rel) const
    {
        note(L'O', rel);
        return moveFile(join(top, rel), join(old, rel));
    }
    bool in(const std::wstring& rel) const
    {
        note(L'N', rel);
        return moveFile(join(tree, rel), join(top, rel));
    }
    // every journalled move undone, newest first; a move that never happened has nothing to undo
    void rollback() const
    {
        std::string text;
        if (!readText(journal, text))
            return;
        const auto all = lines(text);
        int undone = 0, lost = 0;
        for (auto it = all.rbegin(); it != all.rend(); ++it)
        {
            if (it->size() < 3 || (*it)[1] != '\t')
                continue;
            const std::wstring rel = wideOf(it->substr(2));
            const bool newFile = (*it)[0] == 'N';
            const std::wstring from = join(newFile ? top : old, rel), to = join(newFile ? tree : top, rel);
            if (!isFile(from))
                continue;
            ensureParent(to);
            if (MoveFileExW(from.c_str(), to.c_str(), 0))
                undone++;
            else
            {
                lost++;
                logLine(L"rollback: \"" + from + L"\" stays (" + lastError() + L")");
            }
        }
        logLine(L"rolled back " + std::to_wstring(undone) + L" moves" +
                (lost ? L", " + std::to_wstring(lost) + L" could not move back" : L""));
        DeleteFileW(journal.c_str());
    }
};

std::wstring imageOf(HANDLE h)
{
    wchar_t img[2 * MAX_PATH]{};
    DWORD n = 2 * MAX_PATH;
    return QueryFullProcessImageNameW(h, 0, img, &n) ? std::wstring(img, n) : std::wstring();
}

std::vector<DWORD> runningFrom(const std::wstring& prefix)
{
    std::vector<DWORD> pids;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE)
        return pids;
    PROCESSENTRY32W pe{};
    pe.dwSize = sizeof pe;
    for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe))
        if (HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pe.th32ProcessID))
        {
            if (lower(imageOf(h)).rfind(prefix, 0) == 0)
                pids.push_back(pe.th32ProcessID);
            CloseHandle(h);
        }
    CloseHandle(snap);
    return pids;
}

// every process still running an exe from inside the app folder once the app has exited is the app's own (a resident
// render host, a Live host, an encoder): ten seconds to end by itself, then it is ended, so none holds a file the swap
// moves
void endAppProcesses(const std::wstring& top)
{
    const std::wstring prefix = lower(top) + L"\\";
    for (int i = 0; i < 40 && !runningFrom(prefix).empty(); i++)
        Sleep(250);
    for (const DWORD p : runningFrom(prefix))
        if (HANDLE h = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, p))
        {
            logLine(L"ending \"" + imageOf(h) + L"\" (pid " + std::to_wstring(p) +
                    L"), still running from the app folder");
            if (TerminateProcess(h, 1))
                WaitForSingleObject(h, 5000);
            CloseHandle(h);
        }
}

void startApp(const std::wstring& exe, const std::wstring& cwd)
{
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    std::wstring cmd = L"\"" + exe + L"\"";
    if (!CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, cwd.c_str(), &si, &pi))
    {
        logLine(L"the app did not start: " + lastError());
        return;
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
}

[[noreturn]] void fail(const std::wstring& why, const std::wstring& top, bool relaunch, const std::wstring& exe)
{
    logLine(L"no swap: " + why);
    const std::wstring upd = join(top, L"update");
    CreateDirectoryW(upd.c_str(), nullptr);
    HANDLE h = CreateFileW(join(upd, L"failed.txt").c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h != INVALID_HANDLE_VALUE)
    {
        writeAll(h, utf8Of(why));
        CloseHandle(h);
    }
    if (relaunch)
        startApp(join(top, exe), top);
    ExitProcess(1);
}
} // namespace

int WINAPI wWinMain(_In_ HINSTANCE, _In_opt_ HINSTANCE, _In_ PWSTR, _In_ int)
{
    int argc = 0;
    wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv || argc < 6)
        return 2;
    const DWORD pid = wcstoul(argv[1], nullptr, 10);
    const std::wstring top = argv[2], rel = argv[3], exe = argv[5];
    const bool relaunch = argv[4][0] == L'1';
    LocalFree(argv);
    wchar_t tmp[MAX_PATH]{};
    const DWORD tn = GetTempPathW(MAX_PATH, tmp);
    std::wstring temp = tn ? std::wstring(tmp, tn) : std::wstring(L".");
    if (temp.size() > 1 && temp.back() == L'\\')
        temp.pop_back();
    g_log = join(temp, L"smv-update.log");
    SetCurrentDirectoryW(temp.c_str()); // never inside the app folder
    wchar_t triesEnv[16]{};
    g_tries = GetEnvironmentVariableW(L"SMV_UPDATE_TRIES", triesEnv, 16) ? _wtoi(triesEnv) : 30;
    if (g_tries < 1)
        g_tries = 30;
    logLine(L"apply \"" + rel + L"\" into \"" + top + L"\" after pid " + std::to_wstring(pid));
    if (HANDLE hp = OpenProcess(SYNCHRONIZE, FALSE, pid))
    {
        WaitForSingleObject(hp, 10 * 60 * 1000); // the app's exit, ten minutes at most
        CloseHandle(hp);
    }
    endAppProcesses(top);
    // rel = update\<version>\x\<name>: the version's folder holds the journal and the old files
    const size_t s1 = rel.find(L'\\'), s2 = s1 == std::wstring::npos ? s1 : rel.find(L'\\', s1 + 1);
    if (lower(rel.substr(0, s1 == std::wstring::npos ? 0 : s1)) != L"update" || s2 == std::wstring::npos)
        fail(L"the new version's folder is not under update\\<version>: " + rel, top, relaunch, exe);
    Swap sw;
    sw.top = top;
    sw.tree = join(top, rel);
    const std::wstring verDir = join(top, rel.substr(0, s2));
    sw.old = join(verDir, L"old");
    sw.journal = join(verDir, L"journal.txt");
    if (isFile(sw.journal))
    {
        logLine(L"an earlier swap stopped part-way: rolling it back first");
        sw.rollback();
    }
    if (!isFile(join(sw.tree, exe)) || !isFile(join(sw.tree, kAsar)))
        fail(L"the new version's folder holds no " + exe, top, relaunch, exe);
    std::vector<std::wstring> oldList, newList;
    if (!readList(join(top, kList), oldList))
        fail(L"this copy has no release-files.txt", top, relaunch, exe);
    if (!readList(join(sw.tree, kList), newList))
        fail(L"the new version has no release-files.txt", top, relaunch, exe);
    std::unordered_set<std::wstring> oldSet;
    for (const auto& r : oldList)
        oldSet.insert(lower(r));
    const std::wstring finals[] = {kAsar, exe}; // moved last: until then the old exe still starts the old version
    auto isFinal = [&](const std::wstring& r) { return lower(r) == lower(finals[0]) || lower(r) == lower(finals[1]); };
    for (const auto& r : newList)
        if (!isFinal(r) && !isFile(join(sw.tree, r)))
            fail(L"the new version's folder lacks " + r, top, relaunch, exe);
    auto abort = [&](const std::wstring& why) {
        logLine(L"a move failed, rolling back");
        sw.rollback();
        fail(why, top, relaunch, exe);
    };
    int outs = 0, ins = 0;
    for (const auto& r : oldList) // the old version's files leave
        if (!isFinal(r) && isFile(join(top, r)))
        {
            if (!sw.out(r))
                abort(L"the app folder stayed in use (" + r + L")");
            outs++;
        }
    for (const auto& r : newList) // a file of the user's where the new version now ships one gives way to it
        if (!isFinal(r) && !oldSet.count(lower(r)) && isFile(join(top, r)))
        {
            if (!sw.out(r))
                abort(L"the app folder stayed in use (" + r + L")");
            outs++;
        }
    for (const auto& r : newList)
        if (!isFinal(r))
        {
            if (!sw.in(r))
                abort(L"the new version did not move into place (" + r + L")");
            ins++;
        }
    for (const auto& r : finals)
        if (isFile(join(top, r)))
        {
            if (!sw.out(r))
                abort(L"the app folder stayed in use (" + r + L")");
            outs++;
        }
    for (const auto& r : finals)
    {
        if (!sw.in(r))
            abort(L"the new version did not move into place (" + r + L")");
        ins++;
    }
    // folders the old version's files leave empty go (a folder holding anything else stays)
    std::vector<std::wstring> dirs;
    for (const auto& r : oldList)
        for (size_t i = r.find(L'\\'); i != std::wstring::npos; i = r.find(L'\\', i + 1))
            dirs.push_back(r.substr(0, i));
    std::sort(dirs.begin(), dirs.end(), [](const std::wstring& a, const std::wstring& b) {
        return a.size() != b.size() ? a.size() > b.size() : a < b; // the deepest first, equal names side by side
    });
    dirs.erase(std::unique(dirs.begin(), dirs.end()), dirs.end());
    for (const auto& d : dirs)
        RemoveDirectoryW(join(top, d).c_str());
    DeleteFileW(sw.journal.c_str());
    logLine(L"swapped: " + std::to_wstring(outs) + L" files out, " + std::to_wstring(ins) + L" in");
    if (relaunch)
        startApp(join(top, exe), top);
    return 0;
}
