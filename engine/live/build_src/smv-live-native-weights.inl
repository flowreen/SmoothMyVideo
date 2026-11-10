// part of smv-live-native.inl (included there, in place); not a standalone translation unit
// ---- Restore on the host's own kernels -------------------------------------------------------------------------
// The bundled Real-ESRGAN (realesr-animevideov3, BSD-3, engine\realesrgan\realesr-animevideov3.pkl) runs as
// k_restConv0, 16 x k_restConvMid and k_restConvLast. The handoff
// names it as restorePath = kRestNative; weights that are missing or not this model = no Restore (offline: the pass
// is dropped with its line).
static const char* const kRestNative = "native:realesr-animevideov3";
// a --restore-model file runs on the host's kernels in AnimeJaNai's SPAN-F3 form (64 features, ajNatLoad) only; any
// other graph = no Restore, the same way (ajNatWanted's line names why)
static const char* const kRestNativeAj = "native:animejanai";
// --restore-anime4k: Anime4K's Restore CNN UL on the Anime4K module's passes (kA4kRestWeights, no file to look up); a
// --restore-model on the same command line wins over it
static const char* const kRestNativeA4k = "native:anime4k";

// fp32 -> fp16 on the CPU, round to nearest even (torch's .half())
static uint16_t f2hRne(float f)
{
    uint32_t x = 0;
    memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u, ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u)
        return (uint16_t)(sign | (ax > 0x7f800000u ? 0x7e00u : 0x7c00u));
    if (ax >= 0x477ff000u) // rounds to 65520 or more
        return (uint16_t)(sign | 0x7c00u);
    if (ax < 0x38800000u) // below 2^-14: a subnormal half = the value x 2^24 rounded to an even integer
    {
        float v = 0.0f;
        memcpy(&v, &ax, sizeof v);
        return (uint16_t)(sign | (uint32_t)std::nearbyint(v * 16777216.0f));
    }
    const uint32_t r = ax - 0x38000000u;
    return (uint16_t)(sign | ((r + 0xfffu + ((r >> 13) & 1u)) >> 13));
}

static float h2fHost(uint16_t h)
{
    const uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    float f = e == 0 ? std::ldexp((float)m, -24)
                     : (e == 31 ? (m ? NAN : INFINITY) : std::ldexp((float)(m | 0x400u), (int)e - 25));
    return (h & 0x8000u) ? -f : f;
}

// torch.save's file = a stored zip: <dir>/data.pkl, a protocol-2 pickle of the saved object (each tensor =
// torch._utils._rebuild_tensor_v2(its persistent storage ('storage', <dtype class>, key, location, numel), offset,
// size, stride, ...)), and <dir>/data/<key> = each storage's little-endian bytes. Read: the opcodes a state dict uses.
struct PkObj
{
    enum Kind
    {
        None,
        Int,
        Str,
        Tuple,
        List,
        Dict,
        Global,
        Storage,
        Tensor,
        Other
    } k = None;
    int64_t i = 0;
    std::string s;                         // Str; Global = "module.name"; Storage / Tensor = the storage key
    std::vector<std::shared_ptr<PkObj>> v; // Tuple / List items; Dict = key, value, key, value, ...
    int bytes = 0;                         // Storage / Tensor: 4 = fp32, 2 = fp16
    int64_t off = 0;
    std::vector<int64_t> size, stride;
};
using PkP = std::shared_ptr<PkObj>;

static PkP pkLoad(const std::string& b, std::string& why)
{
    std::vector<PkP> st;
    std::vector<size_t> marks;
    std::map<uint32_t, PkP> memo;
    size_t p = 0;
    auto mk = [](PkObj::Kind k) {
        auto o = std::make_shared<PkObj>();
        o->k = k;
        return o;
    };
    auto need = [&](size_t n) { return p + n <= b.size(); };
    auto le = [&](size_t n) {
        uint64_t v = 0;
        for (size_t j = 0; j < n; j++)
            v |= (uint64_t)(unsigned char)b[p + j] << (8 * j);
        p += n;
        return v;
    };
    auto popMark = [&](std::vector<PkP>& items) {
        if (marks.empty() || marks.back() > st.size())
            return false;
        items.assign(st.begin() + (ptrdiff_t)marks.back(), st.end());
        st.resize(marks.back());
        marks.pop_back();
        return true;
    };
    while (p < b.size())
    {
        const unsigned char op = (unsigned char)b[p++];
        bool ok = true;
        switch (op)
        {
        case 0x80: // PROTO
            ok = need(1);
            p += 1;
            break;
        case 0x95: // FRAME
            ok = need(8);
            p += 8;
            break;
        case '}':
            st.push_back(mk(PkObj::Dict));
            break;
        case ']':
            st.push_back(mk(PkObj::List));
            break;
        case ')':
            st.push_back(mk(PkObj::Tuple));
            break;
        case 'N':
            st.push_back(mk(PkObj::None));
            break;
        case 0x88:
        case 0x89: {
            auto o = mk(PkObj::Int);
            o->i = op == 0x88;
            st.push_back(o);
            break;
        }
        case 'K':
        case 'M':
        case 'J': {
            const size_t n = op == 'K' ? 1 : op == 'M' ? 2 : 4;
            if (!(ok = need(n)))
                break;
            const uint64_t u = le(n);
            auto o = mk(PkObj::Int);
            o->i = op == 'J' ? (int64_t)(int32_t)(uint32_t)u : (int64_t)u;
            st.push_back(o);
            break;
        }
        case 0x8a: // LONG1
        {
            if (!(ok = need(1)))
                break;
            const size_t n = (unsigned char)b[p++];
            if (!(ok = n <= 8 && need(n)))
                break;
            const uint64_t u = le(n);
            auto o = mk(PkObj::Int);
            o->i = (n && n < 8 && ((u >> (8 * n - 1)) & 1u)) ? (int64_t)(u | (~0ull << (8 * n))) : (int64_t)u;
            st.push_back(o);
            break;
        }
        case 'X':  // BINUNICODE
        case 0x8c: // SHORT_BINUNICODE
        {
            const size_t ln = op == 'X' ? 4 : 1;
            if (!(ok = need(ln)))
                break;
            const size_t n = (size_t)le(ln);
            if (!(ok = need(n)))
                break;
            auto o = mk(PkObj::Str);
            o->s = b.substr(p, n);
            p += n;
            st.push_back(o);
            break;
        }
        case 'c': // GLOBAL module\nname\n
        {
            const size_t e1 = b.find('\n', p);
            const size_t e2 = e1 == std::string::npos ? e1 : b.find('\n', e1 + 1);
            if (!(ok = e2 != std::string::npos))
                break;
            auto o = mk(PkObj::Global);
            o->s = b.substr(p, e1 - p) + "." + b.substr(e1 + 1, e2 - e1 - 1);
            p = e2 + 1;
            st.push_back(o);
            break;
        }
        case '(':
            marks.push_back(st.size());
            break;
        case 't': {
            std::vector<PkP> it;
            if (!(ok = popMark(it)))
                break;
            auto o = mk(PkObj::Tuple);
            o->v = it;
            st.push_back(o);
            break;
        }
        case 0x85: // TUPLE1 .. TUPLE3
        case 0x86:
        case 0x87: {
            const size_t n = (size_t)(op - 0x84);
            if (!(ok = st.size() >= n))
                break;
            auto o = mk(PkObj::Tuple);
            o->v.assign(st.end() - (ptrdiff_t)n, st.end());
            st.resize(st.size() - n);
            st.push_back(o);
            break;
        }
        case 'q': // BINPUT, LONG_BINPUT
        case 'r': {
            const size_t n = op == 'q' ? 1 : 4;
            if (!(ok = need(n) && !st.empty()))
                break;
            memo[(uint32_t)le(n)] = st.back();
            break;
        }
        case 0x94: // MEMOIZE
            if ((ok = !st.empty()))
                memo[(uint32_t)memo.size()] = st.back();
            break;
        case 'h': // BINGET, LONG_BINGET
        case 'j': {
            const size_t n = op == 'h' ? 1 : 4;
            if (!(ok = need(n)))
                break;
            const auto it = memo.find((uint32_t)le(n));
            if (!(ok = it != memo.end()))
                break;
            st.push_back(it->second);
            break;
        }
        case 'Q': // BINPERSID: ('storage', <dtype class>, key, location, numel)
        {
            if (!(ok = !st.empty()))
                break;
            const PkP id = st.back();
            st.pop_back();
            if (!(ok = id->k == PkObj::Tuple && id->v.size() >= 3 && id->v[1]->k == PkObj::Global &&
                       id->v[2]->k == PkObj::Str))
                break;
            auto o = mk(PkObj::Storage);
            o->s = id->v[2]->s;
            o->bytes = id->v[1]->s == "torch.FloatStorage" ? 4 : id->v[1]->s == "torch.HalfStorage" ? 2 : 0;
            st.push_back(o);
            break;
        }
        case 'R': // REDUCE
        {
            if (!(ok = st.size() >= 2))
                break;
            const PkP args = st.back();
            st.pop_back();
            const PkP fn = st.back();
            st.pop_back();
            PkP o;
            if (fn->k == PkObj::Global && fn->s == "torch._utils._rebuild_tensor_v2" && args->k == PkObj::Tuple &&
                args->v.size() >= 4 && args->v[0]->k == PkObj::Storage && args->v[2]->k == PkObj::Tuple &&
                args->v[3]->k == PkObj::Tuple)
            {
                o = mk(PkObj::Tensor);
                o->s = args->v[0]->s;
                o->bytes = args->v[0]->bytes;
                o->off = args->v[1]->i;
                for (const PkP& d : args->v[2]->v)
                    o->size.push_back(d->i);
                for (const PkP& d : args->v[3]->v)
                    o->stride.push_back(d->i);
            }
            else
                o = mk(fn->k == PkObj::Global && fn->s == "collections.OrderedDict" ? PkObj::Dict : PkObj::Other);
            st.push_back(o);
            break;
        }
        case 's': // SETITEM
        {
            if (!(ok = st.size() >= 3))
                break;
            const PkP val = st.back();
            st.pop_back();
            const PkP key = st.back();
            st.pop_back();
            st.back()->v.push_back(key);
            st.back()->v.push_back(val);
            break;
        }
        case 'u': // SETITEMS
        case 'e': // APPENDS
        {
            std::vector<PkP> it;
            if (!(ok = popMark(it) && !st.empty()))
                break;
            for (const PkP& o : it)
                st.back()->v.push_back(o);
            break;
        }
        case 'a': // APPEND
        {
            if (!(ok = st.size() >= 2))
                break;
            const PkP val = st.back();
            st.pop_back();
            st.back()->v.push_back(val);
            break;
        }
        case 'b': // BUILD: the state is not needed
            if ((ok = st.size() >= 2))
                st.pop_back();
            break;
        case '.':
            if (st.empty())
            {
                why = "an empty pickle";
                return nullptr;
            }
            return st.back();
        default: {
            char h[48];
            snprintf(h, sizeof h, "pickle opcode 0x%02x not read", op);
            why = h;
            return nullptr;
        }
        }
        if (!ok)
        {
            why = "a malformed pickle";
            return nullptr;
        }
    }
    why = "the pickle has no end";
    return nullptr;
}

// a stored zip's entries: name -> (data offset, size); false = not one (`why`: a compressed or zip64 entry, ...)
// the stored zip's directory over the file's n bytes at f: each entry's data offset and size by name
static bool pthZip(const unsigned char* f, size_t fn, std::map<std::string, std::pair<size_t, size_t>>& ent,
                   std::string& why)
{
    auto u16 = [&](size_t o) { return (uint32_t)f[o] | ((uint32_t)f[o + 1] << 8); };
    auto u32 = [&](size_t o) { return u16(o) | (u16(o + 2) << 16); };
    size_t eocd = SIZE_MAX;
    for (size_t o = fn >= 22 ? fn - 22 : 0; fn >= 22 && fn - o <= 65557; o--)
    {
        if (u32(o) == 0x06054b50u)
        {
            eocd = o;
            break;
        }
        if (o == 0)
            break;
    }
    if (eocd == SIZE_MAX)
    {
        why = "not a zip";
        return false;
    }
    const uint32_t n = u16(eocd + 10);
    size_t o = u32(eocd + 16);
    for (uint32_t i = 0; i < n; i++)
    {
        if (o + 46 > fn || u32(o) != 0x02014b50u)
        {
            why = "a bad zip directory";
            return false;
        }
        const uint32_t method = u16(o + 10), csize = u32(o + 20), usize = u32(o + 24), nl = u16(o + 28);
        const uint32_t el = u16(o + 30), cl = u16(o + 32), lho = u32(o + 42);
        if (o + 46 + nl > fn)
        {
            why = "a bad zip directory";
            return false;
        }
        const std::string name((const char*)f + o + 46, nl);
        if (method != 0 || csize != usize || csize == 0xffffffffu || lho == 0xffffffffu)
        {
            why = "a compressed or zip64 entry (" + name + ")";
            return false;
        }
        if ((size_t)lho + 30 > fn || u32(lho) != 0x04034b50u)
        {
            why = "a bad zip entry (" + name + ")";
            return false;
        }
        const size_t data = (size_t)lho + 30 + u16(lho + 26) + u16(lho + 28);
        if (data + csize > fn)
        {
            why = "a cut zip entry (" + name + ")";
            return false;
        }
        ent[name] = {data, (size_t)csize};
        o += 46 + (size_t)nl + el + cl;
    }
    return true;
}

// A torch.save file's state dict (BasicSR's {"params": ...} unwrapped) read in place: the file mapped read-only, each
// tensor in t a view of its stored fp32 values (an fp16 or misaligned tensor converted into own), its shape in S; every
// tensor contiguous. The views live as long as the PthFile.
struct PthFile
{
    HANDLE fh = INVALID_HANDLE_VALUE, mh = nullptr;
    const unsigned char* base = nullptr;
    std::map<std::string, std::span<const float>> t;
    std::map<std::string, std::vector<int64_t>> S;
    std::list<std::vector<float>> own;
    PthFile() = default;
    PthFile(const PthFile&) = delete;
    PthFile& operator=(const PthFile&) = delete;
    ~PthFile()
    {
        if (base)
            UnmapViewOfFile(base);
        if (mh)
            CloseHandle(mh);
        if (fh != INVALID_HANDLE_VALUE)
            CloseHandle(fh);
    }
};

// false = missing or malformed (`why`)
static bool pthStateDict(const std::wstring& path, PthFile& wf, std::string& why)
{
    wf.fh = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN,
                        nullptr);
    LARGE_INTEGER sz = {};
    if (wf.fh == INVALID_HANDLE_VALUE || !GetFileSizeEx(wf.fh, &sz))
    {
        why = "its weights " + wideToUtf8(path) + " are missing";
        return false;
    }
    if (sz.QuadPart < 22)
    {
        why = "not a zip";
        return false;
    }
    const size_t fn = (size_t)sz.QuadPart;
    wf.mh = CreateFileMappingW(wf.fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    wf.base = wf.mh ? (const unsigned char*)MapViewOfFile(wf.mh, FILE_MAP_READ, 0, 0, 0) : nullptr;
    if (!wf.base)
    {
        why = "its weights " + wideToUtf8(path) + " could not be mapped";
        return false;
    }
    const unsigned char* f = wf.base;
    std::map<std::string, std::pair<size_t, size_t>> ent;
    if (!pthZip(f, fn, ent, why))
        return false;
    std::string pkl, dir;
    for (const auto& e : ent)
        if (e.first.size() > 9 && e.first.compare(e.first.size() - 9, 9, "/data.pkl") == 0)
        {
            dir = e.first.substr(0, e.first.size() - 9);
            pkl.assign((const char*)f + e.second.first, e.second.second);
        }
    if (pkl.empty())
    {
        why = "no data.pkl in its weights";
        return false;
    }
    const PkP top = pkLoad(pkl, why);
    if (!top)
        return false;
    PkP sd = top; // BasicSR's save = {"params": the state dict}
    if (top->k == PkObj::Dict)
        for (size_t i = 0; i + 1 < top->v.size(); i += 2)
            if (top->v[i]->k == PkObj::Str && top->v[i]->s == "params")
                sd = top->v[i + 1];
    if (sd->k != PkObj::Dict)
    {
        why = "no state dict in its weights";
        return false;
    }
    for (size_t i = 0; i + 1 < sd->v.size(); i += 2)
    {
        const PkObj& k = *sd->v[i];
        const PkObj& t = *sd->v[i + 1];
        if (k.k != PkObj::Str || t.k != PkObj::Tensor || (t.bytes != 4 && t.bytes != 2) ||
            t.size.size() != t.stride.size())
        {
            why = "a state dict entry that is not an fp32 / fp16 tensor";
            return false;
        }
        int64_t n = 1, want = 1;
        for (int64_t d : t.size)
            n *= d;
        for (size_t d = t.size.size(); d-- > 0;)
        {
            if (t.size[d] != 1 && t.stride[d] != want)
            {
                why = "a tensor that is not contiguous (" + k.s + ")";
                return false;
            }
            want *= t.size[d];
        }
        const auto e = ent.find(dir + "/data/" + t.s);
        if (n <= 0 || t.off < 0 || e == ent.end() || (uint64_t)(t.off + n) * (uint64_t)t.bytes > e->second.second)
        {
            why = "a tensor's storage is missing or short (" + k.s + ")";
            return false;
        }
        const unsigned char* src = f + e->second.first + (size_t)t.off * (size_t)t.bytes;
        if (t.bytes == 4 && ((uintptr_t)src & 3) == 0)
            wf.t[k.s] = std::span<const float>(reinterpret_cast<const float*>(src), (size_t)n);
        else
        {
            std::vector<float>& v = wf.own.emplace_back((size_t)n);
            for (size_t j = 0; j < (size_t)n; j++)
            {
                if (t.bytes == 4)
                    memcpy(&v[j], src + 4 * j, 4);
                else
                {
                    uint16_t h = 0;
                    memcpy(&h, src + 2 * j, 2);
                    v[j] = h2fHost(h);
                }
            }
            wf.t[k.s] = std::span<const float>(v);
        }
        wf.S[k.s] = t.size;
    }
    return true;
}

// The bundled model's weights for the kernels, in one block (restNatOff's offsets, 256-byte aligned): k_restConv0's
// w0[64][27] and b0 fp32 + its slopes fp16; per middle layer its fragment-order weights wf[tap][kc][q 4][lane][8 halves]
// (lane's B registers for n-tiles 2q, 2q + 1 at channels 16 kc .. + 15: n = 16 q + lane / 4 (+ 8), k = 2 (lane % 4)
// (+ 1, + 8, + 9)), bias and slopes fp16; the last layer's wf[tap][kc][q 3][lane][8] and bias fp16. Every value is
// first rounded to fp16, as the model runs (.half()). false = missing, or not this model (`why`).
static std::vector<uint16_t> restFragOf(std::span<const float> w, int nout, int cin, int coff, int taps);
static bool restNatLoad(std::vector<uint8_t>& blob, std::vector<size_t>& off, std::string& why)
{
    off.assign(3 + 16 * 3 + 2, 0);
    PthFile pf;
    if (!pthStateDict(g_nativeEngineDir + L"\\realesrgan\\realesr-animevideov3.pkl", pf, why))
        return false;
    auto& wts = pf.t;
    const auto& S = pf.S;
    // the form: body.0 conv 3 -> 64, body.1 PReLU 64, 16 x (conv 64 -> 64, PReLU 64), body.34 conv 64 -> 48, 3 x 3
    auto has = [&](const std::string& n, const std::vector<int64_t>& s) {
        const auto it = S.find(n);
        return it != S.end() && it->second == s;
    };
    bool form = wts.size() == 53 && has("body.0.weight", {64, 3, 3, 3}) && has("body.0.bias", {64}) &&
                has("body.1.weight", {64}) && has("body.34.weight", {48, 64, 3, 3}) && has("body.34.bias", {48});
    for (int L = 0; L < 16 && form; L++)
    {
        const std::string c = "body." + std::to_string(2 + 2 * L), a = "body." + std::to_string(3 + 2 * L);
        form = has(c + ".weight", {64, 64, 3, 3}) && has(c + ".bias", {64}) && has(a + ".weight", {64});
    }
    if (!form)
    {
        why = "its weights are not realesr-animevideov3's form";
        return false;
    }
    blob.clear();
    auto put = [&](const void* p, size_t n) {
        const size_t o = (blob.size() + 255) & ~(size_t)255;
        blob.resize(o + n);
        memcpy(blob.data() + o, p, n);
        return o;
    };
    auto halves = [](std::span<const float> v) {
        std::vector<uint16_t> h(v.size());
        for (size_t i = 0; i < v.size(); i++)
            h[i] = f2hRne(v[i]);
        return h;
    };
    // conv 0 keeps fp32 storage of the fp16-rounded values; every other weight goes through f2hRne once (rounding an
    // fp16 value again changes nothing)
    auto rounded = [](std::span<const float> v) {
        std::vector<float> r(v.begin(), v.end());
        for (float& x : r)
            x = h2fHost(f2hRne(x));
        return r;
    };
    auto fragOf = [](std::span<const float> w, int nout) { return restFragOf(w, nout, 64, 0, 9); };
    const std::vector<float> w0 = rounded(wts["body.0.weight"]), b0 = rounded(wts["body.0.bias"]);
    off[0] = put(w0.data(), w0.size() * sizeof(float));
    off[1] = put(b0.data(), b0.size() * sizeof(float));
    const std::vector<uint16_t> s0 = halves(wts["body.1.weight"]);
    off[2] = put(s0.data(), s0.size() * 2);
    for (int L = 0; L < 16; L++)
    {
        const std::string c = "body." + std::to_string(2 + 2 * L), a = "body." + std::to_string(3 + 2 * L);
        const std::vector<uint16_t> wf = fragOf(wts[c + ".weight"], 64), bs = halves(wts[c + ".bias"]),
                                    sl = halves(wts[a + ".weight"]);
        off[3 + 3 * L] = put(wf.data(), wf.size() * 2);
        off[4 + 3 * L] = put(bs.data(), bs.size() * 2);
        off[5 + 3 * L] = put(sl.data(), sl.size() * 2);
    }
    const std::vector<uint16_t> wl = fragOf(wts["body.34.weight"], 48), bl = halves(wts["body.34.bias"]);
    off[51] = put(wl.data(), wl.size() * 2);
    off[52] = put(bl.data(), bl.size() * 2);
    return true;
}

// a conv's fp16 fragment-order weights for the tensor-core kernels: w[o][cin][taps] (PyTorch's layout) -> per tap, per
// 16-channel step kc of the 64 channels from coff, per n-tile pair q of nout, per lane the 8 halves of its B registers
// (n = 16 q + lane / 4 (+ 8), k = 2 (lane % 4) (+ 1, + 8, + 9))
static std::vector<uint16_t> restFragOf(std::span<const float> w, int nout, int cin, int coff, int taps)
{
    const int nq = nout / 16;
    std::vector<uint16_t> fr((size_t)taps * 4 * nq * 32 * 8);
    size_t i = 0;
    for (int t = 0; t < taps; t++)
        for (int kc = 0; kc < 4; kc++)
            for (int q = 0; q < nq; q++)
                for (int l = 0; l < 32; l++)
                {
                    const int n0 = 16 * q + (l >> 2), c0 = coff + 16 * kc + 2 * (l & 3);
                    const int ns[8] = {n0, n0, n0, n0, n0 + 8, n0 + 8, n0 + 8, n0 + 8};
                    const int cs[8] = {c0, c0 + 1, c0 + 8, c0 + 9, c0, c0 + 1, c0 + 8, c0 + 9};
                    for (int j = 0; j < 8; j++)
                        fr[i++] = f2hRne(w[((size_t)ns[j] * cin + (size_t)cs[j]) * taps + (size_t)t]);
                }
    return fr;
}

// the native route for this session: the bundled model (no --restore-model), its weights readable
static bool restNatWanted()
{
    if (!g_restoreModel.empty())
        return false;
    static int ok = -1; // the file is read once a process
    if (ok < 0)
    {
        std::vector<uint8_t> blob;
        std::vector<size_t> off;
        std::string why;
        ok = restNatLoad(blob, off, why) ? 1 : 0;
        if (!ok)
            LOG("native: Restore unavailable (%s)\n", why.c_str());
    }
    return ok == 1;
}

// one protobuf field: its number, wire type and value (a varint / fixed one, or the bytes of a length-delimited one)
struct PbField
{
    uint32_t num = 0, wire = 0;
    uint64_t v = 0;
    const uint8_t* p = nullptr;
    size_t n = 0;
};
static bool pbVarint(const uint8_t*& p, const uint8_t* end, uint64_t& v)
{
    v = 0;
    for (int s = 0; s < 64 && p < end; s += 7)
    {
        const uint8_t b = *p++;
        v |= (uint64_t)(b & 0x7fu) << s;
        if (!(b & 0x80u))
            return true;
    }
    return false;
}
static bool pbNext(const uint8_t*& p, const uint8_t* end, PbField& f)
{
    uint64_t key = 0;
    if (p >= end || !pbVarint(p, end, key))
        return false;
    f.num = (uint32_t)(key >> 3);
    f.wire = (uint32_t)(key & 7u);
    f.v = 0;
    f.p = nullptr;
    f.n = 0;
    const size_t left = (size_t)(end - p);
    switch (f.wire)
    {
    case 0:
        return pbVarint(p, end, f.v);
    case 1:
        if (left < 8)
            return false;
        memcpy(&f.v, p, 8);
        p += 8;
        return true;
    case 5: {
        if (left < 4)
            return false;
        uint32_t u = 0;
        memcpy(&u, p, 4);
        f.v = u;
        p += 4;
        return true;
    }
    case 2: {
        uint64_t n = 0;
        if (!pbVarint(p, end, n) || n > (uint64_t)(end - p))
            return false;
        f.p = p;
        f.n = (size_t)n;
        p += f.n;
        return true;
    }
    default:
        return false;
    }
}

// the ONNX subset a conv net's weights and wiring need: the graph's nodes (op, inputs, outputs, int / string
// attributes) and its float / fp16 initializers as floats (shape constants: their dims only)
struct OnnxNode
{
    std::string op;
    std::vector<std::string> in, out;
    std::map<std::string, std::vector<int64_t>> ints;
    std::map<std::string, std::string> strs;
};
struct OnnxInit
{
    std::vector<int64_t> dims;
    std::vector<float> f;
};
struct OnnxGraph
{
    std::vector<OnnxNode> nodes;
    std::map<std::string, OnnxInit> inits;
    std::vector<std::string> inputs, outputs;
};

static bool onnxRead(const std::vector<char>& file, OnnxGraph& g, std::string& why)
{
    const uint8_t* p = (const uint8_t*)file.data();
    const uint8_t* end = p + file.size();
    PbField f;
    const uint8_t *gp = nullptr, *ge = nullptr;
    while (pbNext(p, end, f))
        if (f.num == 7 && f.wire == 2) // ModelProto.graph
        {
            gp = f.p;
            ge = f.p + f.n;
        }
    if (!gp)
    {
        why = "no ONNX graph";
        return false;
    }
    auto ints = [](const PbField& x, std::vector<int64_t>& v) { // repeated int64: packed or one at a time
        if (x.wire == 0)
            v.push_back((int64_t)x.v);
        else if (x.wire == 2)
        {
            const uint8_t* q = x.p;
            uint64_t u = 0;
            while (q < x.p + x.n && pbVarint(q, x.p + x.n, u))
                v.push_back((int64_t)u);
        }
    };
    auto name = [](const PbField& x) { return std::string((const char*)x.p, x.n); };
    while (pbNext(gp, ge, f))
    {
        if (f.wire != 2)
            continue;
        const uint8_t *q = f.p, *qe = f.p + f.n;
        PbField a;
        if (f.num == 1) // NodeProto
        {
            OnnxNode nd;
            while (pbNext(q, qe, a))
                if (a.num == 1 && a.wire == 2)
                    nd.in.push_back(name(a));
                else if (a.num == 2 && a.wire == 2)
                    nd.out.push_back(name(a));
                else if (a.num == 4 && a.wire == 2)
                    nd.op = name(a);
                else if (a.num == 5 && a.wire == 2) // AttributeProto: name 1, i 3, s 4, ints 8
                {
                    const uint8_t *r = a.p, *re = a.p + a.n;
                    PbField b;
                    std::string an;
                    std::vector<int64_t> iv;
                    std::string sv;
                    bool hasS = false;
                    while (pbNext(r, re, b))
                        if (b.num == 1 && b.wire == 2)
                            an = name(b);
                        else if (b.num == 3 && b.wire == 0)
                            iv.push_back((int64_t)b.v);
                        else if (b.num == 8)
                            ints(b, iv);
                        else if (b.num == 4 && b.wire == 2)
                        {
                            sv = name(b);
                            hasS = true;
                        }
                    if (hasS)
                        nd.strs[an] = sv;
                    else
                        nd.ints[an] = iv;
                }
            g.nodes.push_back(std::move(nd));
        }
        else if (f.num == 5) // TensorProto: dims 1, data_type 2, float_data 4, int32_data 5, name 8, raw_data 9,
        {                    // data_location 14 (1 = external)
            OnnxInit t;
            std::string tn;
            int64_t type = 0, loc = 0;
            const uint8_t* raw = nullptr;
            size_t rawN = 0;
            std::vector<int64_t> i32;
            std::vector<float> f32;
            while (pbNext(q, qe, a))
                if (a.num == 1)
                    ints(a, t.dims);
                else if (a.num == 2 && a.wire == 0)
                    type = (int64_t)a.v;
                else if (a.num == 8 && a.wire == 2)
                    tn = name(a);
                else if (a.num == 9 && a.wire == 2)
                {
                    raw = a.p;
                    rawN = a.n;
                }
                else if (a.num == 5)
                    ints(a, i32);
                else if (a.num == 4 && a.wire == 2)
                    for (size_t j = 0; j + 4 <= a.n; j += 4)
                    {
                        float v = 0.0f;
                        memcpy(&v, a.p + j, 4);
                        f32.push_back(v);
                    }
                else if (a.num == 4 && a.wire == 5)
                {
                    float v = 0.0f;
                    const uint32_t u = (uint32_t)a.v;
                    memcpy(&v, &u, 4);
                    f32.push_back(v);
                }
                else if (a.num == 14 && a.wire == 0)
                    loc = (int64_t)a.v;
            if (loc == 1)
            {
                why = "an initializer in an external file (" + tn + ")";
                return false;
            }
            if (type == 1) // FLOAT
            {
                if (raw)
                    for (size_t j = 0; j + 4 <= rawN; j += 4)
                    {
                        float v = 0.0f;
                        memcpy(&v, raw + j, 4);
                        t.f.push_back(v);
                    }
                else
                    t.f = f32;
            }
            else if (type == 10) // FLOAT16: raw bytes, or one value per int32_data entry
            {
                if (raw)
                    for (size_t j = 0; j + 2 <= rawN; j += 2)
                    {
                        uint16_t h = 0;
                        memcpy(&h, raw + j, 2);
                        t.f.push_back(h2fHost(h));
                    }
                else
                    for (int64_t v : i32)
                        t.f.push_back(h2fHost((uint16_t)v));
            }
            g.inits[tn] = std::move(t);
        }
        else if (f.num == 11 || f.num == 12) // ValueInfoProto: name 1
            while (pbNext(q, qe, a))
                if (a.num == 1 && a.wire == 2)
                    (f.num == 11 ? g.inputs : g.outputs).push_back(name(a));
    }
    return true;
}

// AnimeJaNai's SPAN-F3 net (64 features, n blocks) from a --restore-model ONNX, checked node by node along its data
// path: Pad (reflect) -> PixelUnshuffle(2) as Reshape / Transpose [0,1,3,5,2,4] / Reshape -> Conv 12 -> 64 3 x 3 (x0)
// -> per block Conv 3 x 3 -> Relu -> Conv 3 x 3 -> Relu -> Conv 3 x 3 (c3), Add(block input, c3), Conv 1 x 1 (c3) ->
// Tanh, Mul(Add, Tanh) -> Concat [x0, the last block, the first block, the last block's first Relu] -> Conv 1 x 1
// 256 -> 64 -> Conv 3 x 3 64 -> 48 -> DepthToSpace(4, CRD) -> Slices -> the output. Every conv: bias, stride 1, group 1,
// pad 1 (3 x 3) / 0 (1 x 1). Anything else = not this form (`why`). The weights for the kernels, in one block (off: the
// first conv's w[64][108] + b fp32; per block c1 / c2 / c3 fragment-order + bias + slope (0 = ReLU, 1 = none) fp16,
// the 1 x 1's fragments + bias fp32; the concat conv's fragments (4 sources) + bias fp32; the last conv's + bias fp16).
// AnimeJaNai's SPAN-F3 weights by layer, fp32 values (a weight stored as fp16 reads back exactly): conv_1 (64 x 12 x 3
// x 3), per block three 3 x 3 convs and the 1 x 1 attention conv, conv_cat (64 x 256 x 1 x 1), upsampler (48 x 64 x 3
// x 3); the .pkl the app keeps (--convert-restore-model) names them conv_1, block_K.c1 .. c3, block_K.att, conv_cat,
// upsampler (.weight / .bias)
struct AjT
{
    std::vector<float> w, b;
};
struct AjBlock
{
    AjT c[3], att;
};
struct AjWeights
{
    AjT first, cat, last;
    std::vector<AjBlock> blocks;
};

// the weights out of the author's ONNX, after the graph's form check
static bool ajFromOnnx(const std::string& path, AjWeights& aw, std::string& why)
{
    std::vector<char> file;
    if (!lkReadFile(path, file) || file.empty())
    {
        why = "its model file cannot be read";
        return false;
    }
    OnnxGraph g;
    if (!onnxRead(file, g, why))
        return false;
    std::map<std::string, int> prod;
    std::map<std::string, std::vector<int>> cons;
    for (int i = 0; i < (int)g.nodes.size(); i++)
    {
        for (const std::string& o : g.nodes[i].out)
            prod[o] = i;
        for (const std::string& x : g.nodes[i].in)
            cons[x].push_back(i);
    }
    auto bad = [&](const std::string& where) {
        why = "not AnimeJaNai's SPAN-F3 form (" + where + ")";
        return false;
    };
    auto node = [&](const std::string& t) -> const OnnxNode* {
        const auto it = prod.find(t);
        return it == prod.end() ? nullptr : &g.nodes[it->second];
    };
    auto dims = [&](const std::string& t) {
        const auto it = g.inits.find(t);
        return it == g.inits.end() ? std::vector<int64_t>() : it->second.dims;
    };
    // the one consumer of tensor t with op `op` (and, for a Conv, weight dims wd), or -1
    auto take = [&](const std::string& t, const char* op, const std::vector<int64_t>& wd) {
        int hit = -1, n = 0;
        for (int i : cons[t])
            if (g.nodes[i].op == op && (wd.empty() || (g.nodes[i].in.size() == 3 && dims(g.nodes[i].in[1]) == wd)))
            {
                hit = i;
                n++;
            }
        return n == 1 ? hit : -1;
    };
    auto convOk = [&](int i, int k) {
        const OnnxNode& c = g.nodes[i];
        auto at = [&](const char* a, const std::vector<int64_t>& want) {
            const auto it = c.ints.find(a);
            return it == c.ints.end() || it->second == want;
        };
        const std::vector<int64_t> pd(4, k == 3 ? 1 : 0);
        return c.in.size() == 3 && dims(c.in[2]).size() == 1 && at("strides", {1, 1}) && at("dilations", {1, 1}) &&
               at("group", {1}) && at("kernel_shape", {k, k}) && at("pads", pd) && c.out.size() == 1;
    };
    // the head: input -> Pad (reflect) -> Reshape -> Transpose [0,1,3,5,2,4] -> Reshape -> the first conv
    const std::vector<int64_t> w12 = {64, 12, 3, 3}, w64 = {64, 64, 3, 3}, w1 = {64, 64, 1, 1}, wc = {64, 256, 1, 1},
                               wl = {48, 64, 3, 3};
    int first = -1, nFirst = 0;
    for (int i = 0; i < (int)g.nodes.size(); i++)
        if (g.nodes[i].op == "Conv" && g.nodes[i].in.size() == 3 && dims(g.nodes[i].in[1]) == w12)
        {
            first = i;
            nFirst++;
        }
    if (nFirst != 1 || !convOk(first, 3) || g.inputs.empty() || g.outputs.size() != 1)
        return bad("the first conv");
    {
        const OnnxNode* r2 = node(g.nodes[first].in[0]);
        const OnnxNode* tr = r2 && r2->op == "Reshape" ? node(r2->in[0]) : nullptr;
        const OnnxNode* r1 = tr && tr->op == "Transpose" ? node(tr->in[0]) : nullptr;
        const OnnxNode* pad = r1 && r1->op == "Reshape" ? node(r1->in[0]) : nullptr;
        const auto perm = tr ? tr->ints.find("perm") : std::map<std::string, std::vector<int64_t>>::const_iterator();
        const auto mode = pad ? pad->strs.find("mode") : std::map<std::string, std::string>::const_iterator();
        if (!pad || pad->op != "Pad" || pad->in.empty() || pad->in[0] != g.inputs[0] || perm == tr->ints.end() ||
            perm->second != std::vector<int64_t>{0, 1, 3, 5, 2, 4} || mode == pad->strs.end() ||
            mode->second != "reflect")
            return bad("the pad and pixel unshuffle");
    }
    struct Block
    {
        int c[3], att;
        std::string relu1, out;
    };
    std::vector<Block> blocks;
    std::string cur = g.nodes[first].out[0];
    for (;;)
    {
        Block b{};
        b.c[0] = take(cur, "Conv", w64);
        if (b.c[0] < 0)
            break;
        const int r1 = take(g.nodes[b.c[0]].out[0], "Relu", {});
        b.c[1] = r1 < 0 ? -1 : take(g.nodes[r1].out[0], "Conv", w64);
        const int r2 = b.c[1] < 0 ? -1 : take(g.nodes[b.c[1]].out[0], "Relu", {});
        b.c[2] = r2 < 0 ? -1 : take(g.nodes[r2].out[0], "Conv", w64);
        if (b.c[2] < 0 || !convOk(b.c[0], 3) || !convOk(b.c[1], 3) || !convOk(b.c[2], 3))
            return bad("block " + std::to_string(blocks.size() + 1) + "'s convs");
        const std::string c3 = g.nodes[b.c[2]].out[0];
        const int add = take(c3, "Add", {});
        b.att = take(c3, "Conv", w1);
        const int th = b.att < 0 ? -1 : take(g.nodes[b.att].out[0], "Tanh", {});
        const int mul = add < 0 || th < 0 ? -1 : take(g.nodes[add].out[0], "Mul", {});
        if (add < 0 || th < 0 || mul < 0 || !convOk(b.att, 1) || cons[c3].size() != 2 ||
            !((g.nodes[add].in[0] == cur && g.nodes[add].in[1] == c3) ||
              (g.nodes[add].in[1] == cur && g.nodes[add].in[0] == c3)) ||
            std::find(g.nodes[mul].in.begin(), g.nodes[mul].in.end(), g.nodes[th].out[0]) == g.nodes[mul].in.end())
            return bad("block " + std::to_string(blocks.size() + 1) + "'s attention");
        b.relu1 = g.nodes[r1].out[0];
        b.out = cur = g.nodes[mul].out[0];
        blocks.push_back(b);
        if (blocks.size() > 16)
            return bad("more than 16 blocks");
    }
    if (blocks.empty())
        return bad("no block");
    const int cat = take(cur, "Concat", {});
    const std::vector<std::string> want = {g.nodes[first].out[0], blocks.back().out, blocks.front().out,
                                           blocks.back().relu1};
    const int cc = cat < 0 ? -1 : take(g.nodes[cat].out[0], "Conv", wc);
    const int last = cc < 0 ? -1 : take(g.nodes[cc].out[0], "Conv", wl);
    const int d2s = last < 0 ? -1 : take(g.nodes[last].out[0], "DepthToSpace", {});
    if (cat < 0 || g.nodes[cat].in != want || cc < 0 || last < 0 || d2s < 0 || !convOk(cc, 1) || !convOk(last, 3))
        return bad("the tail");
    {
        const OnnxNode& d = g.nodes[d2s];
        const auto bs = d.ints.find("blocksize");
        const auto md = d.strs.find("mode");
        if (bs == d.ints.end() || bs->second != std::vector<int64_t>{4} || (md != d.strs.end() && md->second != "CRD"))
            return bad("the depth to space");
        std::string t = g.outputs[0];
        for (int s = 0; s < 3 && node(t) && node(t)->op == "Slice"; s++)
            t = node(t)->in[0];
        if (t != d.out[0])
            return bad("the output's crop");
    }
    int nconv = 0, nrelu = 0, ntanh = 0;
    for (const OnnxNode& nd : g.nodes)
    {
        nconv += nd.op == "Conv";
        nrelu += nd.op == "Relu";
        ntanh += nd.op == "Tanh";
    }
    const int n = (int)blocks.size();
    if (nconv != 3 + 4 * n || nrelu != 2 * n || ntanh != n)
        return bad("extra layers");
    // the weights
    auto wOf = [&](int i) -> const std::vector<float>& { return g.inits[g.nodes[i].in[1]].f; };
    auto bOf = [&](int i) -> const std::vector<float>& { return g.inits[g.nodes[i].in[2]].f; };
    auto tOf = [&](int i) { return AjT{wOf(i), bOf(i)}; };
    aw.first = tOf(first);
    aw.blocks.clear();
    for (const Block& b : blocks)
        aw.blocks.push_back(AjBlock{{tOf(b.c[0]), tOf(b.c[1]), tOf(b.c[2])}, tOf(b.att)});
    aw.cat = tOf(cc);
    aw.last = tOf(last);
    return true;
}

// the kernels' block (offsets 256-byte aligned) from the weights; false = a weight of the wrong size (`why`)
static bool ajBlob(const AjWeights& aw, std::vector<uint8_t>& blob, std::vector<size_t>& off, int& nblocks,
                   std::string& why)
{
    const int n = (int)aw.blocks.size();
    auto bad = [&](const std::string& where) {
        why = "not AnimeJaNai's SPAN-F3 form (" + where + ")";
        return false;
    };
    if (n < 1 || n > 16)
        return bad("the blocks");
    blob.clear();
    off.assign((size_t)(6 + 11 * n), 0);
    auto put = [&](const void* p, size_t bytes) {
        const size_t o = (blob.size() + 255) & ~(size_t)255;
        blob.resize(o + bytes);
        memcpy(blob.data() + o, p, bytes);
        return o;
    };
    auto halves = [](const std::vector<float>& v) {
        std::vector<uint16_t> h(v.size());
        for (size_t i = 0; i < v.size(); i++)
            h[i] = f2hRne(v[i]);
        return h;
    };
    auto sizesOk = [&](const AjT& t, size_t nw, size_t nb) { return t.w.size() == nw && t.b.size() == nb; };
    if (!sizesOk(aw.first, 64 * 108, 64) || !sizesOk(aw.cat, 64 * 256, 64) || !sizesOk(aw.last, 48 * 64 * 9, 48))
        return bad("a weight's size");
    off[0] = put(aw.first.w.data(), aw.first.w.size() * sizeof(float));
    off[1] = put(aw.first.b.data(), 64 * sizeof(float));
    const std::vector<uint16_t> zeros(64, 0), ones(64, 0x3c00);
    for (int k = 0; k < n; k++)
    {
        const AjBlock& b = aw.blocks[(size_t)k];
        for (int j = 0; j < 3; j++)
        {
            if (!sizesOk(b.c[j], 64 * 64 * 9, 64))
                return bad("a weight's size");
            const std::vector<uint16_t> wf = restFragOf(b.c[j].w, 64, 64, 0, 9), bs = halves(b.c[j].b);
            off[(size_t)(2 + 11 * k + 3 * j)] = put(wf.data(), wf.size() * 2);
            off[(size_t)(3 + 11 * k + 3 * j)] = put(bs.data(), bs.size() * 2);
            off[(size_t)(4 + 11 * k + 3 * j)] = put(j < 2 ? zeros.data() : ones.data(), 64 * 2);
        }
        if (!sizesOk(b.att, 64 * 64, 64))
            return bad("a weight's size");
        const std::vector<uint16_t> wa = restFragOf(b.att.w, 64, 64, 0, 1);
        off[(size_t)(11 + 11 * k)] = put(wa.data(), wa.size() * 2);
        off[(size_t)(12 + 11 * k)] = put(b.att.b.data(), 64 * sizeof(float));
    }
    std::vector<uint16_t> wcat;
    for (int s = 0; s < 4; s++)
    {
        const std::vector<uint16_t> part = restFragOf(aw.cat.w, 64, 256, 64 * s, 1);
        wcat.insert(wcat.end(), part.begin(), part.end());
    }
    off[(size_t)(2 + 11 * n)] = put(wcat.data(), wcat.size() * 2);
    off[(size_t)(3 + 11 * n)] = put(aw.cat.b.data(), 64 * sizeof(float));
    const std::vector<uint16_t> wlf = restFragOf(aw.last.w, 48, 64, 0, 9), bl = halves(aw.last.b);
    off[(size_t)(4 + 11 * n)] = put(wlf.data(), wlf.size() * 2);
    off[(size_t)(5 + 11 * n)] = put(bl.data(), bl.size() * 2);
    nblocks = n;
    return true;
}

// the weights out of the .pkl the converter writes: every name with its exact shape, nothing else
static bool ajFromPkl(const std::wstring& path, AjWeights& aw, std::string& why)
{
    PthFile pf;
    if (!pthStateDict(path, pf, why))
        return false;
    auto bad = [&](const std::string& where) {
        why = "not AnimeJaNai's SPAN-F3 form (" + where + ")";
        return false;
    };
    auto get = [&](const std::string& n, const std::vector<int64_t>& wd, AjT& t) {
        const auto w = pf.S.find(n + ".weight"), b = pf.S.find(n + ".bias");
        if (w == pf.S.end() || b == pf.S.end() || w->second != wd || b->second != std::vector<int64_t>{wd[0]})
            return false;
        const std::span<const float> tw = pf.t.at(n + ".weight"), tb = pf.t.at(n + ".bias");
        t.w.assign(tw.begin(), tw.end());
        t.b.assign(tb.begin(), tb.end());
        return true;
    };
    const std::vector<int64_t> w64 = {64, 64, 3, 3};
    if (!get("conv_1", {64, 12, 3, 3}, aw.first))
        return bad("conv_1");
    aw.blocks.clear();
    for (int k = 1; k <= 16 && pf.S.count("block_" + std::to_string(k) + ".c1.weight"); k++)
    {
        const std::string p = "block_" + std::to_string(k);
        AjBlock b;
        if (!get(p + ".c1", w64, b.c[0]) || !get(p + ".c2", w64, b.c[1]) || !get(p + ".c3", w64, b.c[2]) ||
            !get(p + ".att", {64, 64, 1, 1}, b.att))
            return bad(p);
        aw.blocks.push_back(std::move(b));
    }
    if (aw.blocks.empty())
        return bad("no block");
    if (!get("conv_cat", {64, 256, 1, 1}, aw.cat) || !get("upsampler", {48, 64, 3, 3}, aw.last))
        return bad("the tail");
    if (pf.t.size() != 2 * (3 + 4 * aw.blocks.size()))
        return bad("extra tensors");
    return true;
}

// the .pkl (torch.save's zip, stored entries: archive/data.pkl + archive/data/<i>, version, byteorder): an OrderedDict
// of _rebuild_tensor_v2 tensors, each fp16 when every value round-trips exactly (the author's fp16 ONNX: all), else
// fp32; pthStateDict and torch.load read it; no timestamps, so one ONNX always gives the same bytes
static bool ajWritePkl(const AjWeights& aw, const std::wstring& path, std::string& why)
{
    struct Ent
    {
        std::string name;
        const std::vector<float>* v;
        std::vector<int64_t> dims;
    };
    std::vector<Ent> ts;
    auto add = [&](const std::string& n, const AjT& t, const std::vector<int64_t>& wd) {
        ts.push_back({n + ".weight", &t.w, wd});
        ts.push_back({n + ".bias", &t.b, {wd[0]}});
    };
    add("conv_1", aw.first, {64, 12, 3, 3});
    for (size_t k = 0; k < aw.blocks.size(); k++)
    {
        const std::string p = "block_" + std::to_string(k + 1);
        for (int j = 0; j < 3; j++)
            add(p + ".c" + std::to_string(j + 1), aw.blocks[k].c[j], {64, 64, 3, 3});
        add(p + ".att", aw.blocks[k].att, {64, 64, 1, 1});
    }
    add("conv_cat", aw.cat, {64, 256, 1, 1});
    add("upsampler", aw.last, {48, 64, 3, 3});
    std::string p = "\x80\x02";
    auto glob = [&](const char* m, const char* n) {
        p += 'c';
        p += m;
        p += '\n';
        p += n;
        p += '\n';
    };
    auto str = [&](const std::string& s) {
        const uint32_t l = (uint32_t)s.size();
        p += 'X';
        p.append((const char*)&l, 4);
        p += s;
    };
    auto i32 = [&](int64_t v) {
        const int32_t x = (int32_t)v;
        p += 'J';
        p.append((const char*)&x, 4);
    };
    glob("collections", "OrderedDict");
    p += ")R(";
    std::vector<std::string> data;
    for (size_t i = 0; i < ts.size(); i++)
    {
        const std::vector<float>& v = *ts[i].v;
        int64_t numel = 1;
        for (int64_t d : ts[i].dims)
            numel *= d;
        if (numel != (int64_t)v.size())
        {
            why = "a weight's size (" + ts[i].name + ")";
            return false;
        }
        bool half = true;
        for (float x : v)
            if (h2fHost(f2hRne(x)) != x)
            {
                half = false;
                break;
            }
        std::string d;
        if (half)
        {
            d.resize(v.size() * 2);
            for (size_t j = 0; j < v.size(); j++)
            {
                const uint16_t h = f2hRne(v[j]);
                memcpy(&d[2 * j], &h, 2);
            }
        }
        else
            d.assign((const char*)v.data(), v.size() * sizeof(float));
        data.push_back(std::move(d));
        str(ts[i].name);
        glob("torch._utils", "_rebuild_tensor_v2");
        p += "((";
        str("storage");
        glob("torch", half ? "HalfStorage" : "FloatStorage");
        str(std::to_string(i));
        str("cpu");
        i32(numel);
        p += "tQ";
        i32(0);
        p += '(';
        for (int64_t dd : ts[i].dims)
            i32(dd);
        p += "t(";
        std::vector<int64_t> stride(ts[i].dims.size(), 1);
        for (size_t s = stride.size(); s-- > 1;)
            stride[s - 1] = stride[s] * ts[i].dims[s];
        for (int64_t s : stride)
            i32(s);
        p += "t\x89";
        glob("collections", "OrderedDict");
        p += ")RtR";
    }
    p += "u.";
    std::vector<std::pair<std::string, std::string>> files = {
        {"archive/data.pkl", p}, {"archive/byteorder", "little"}, {"archive/version", "3\n"}};
    for (size_t i = 0; i < data.size(); i++)
        files.push_back({"archive/data/" + std::to_string(i), data[i]});
    static uint32_t crcTab[256];
    if (!crcTab[1])
        for (uint32_t i = 0; i < 256; i++)
        {
            uint32_t c = i;
            for (int k = 0; k < 8; k++)
                c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            crcTab[i] = c;
        }
    auto crc32 = [](const std::string& s) {
        uint32_t c = 0xFFFFFFFFu;
        for (unsigned char ch : s)
            c = crcTab[(c ^ ch) & 0xFF] ^ (c >> 8);
        return c ^ 0xFFFFFFFFu;
    };
    std::string z, cd;
    auto le = [](std::string& o, uint32_t v, int bytes) {
        for (int i = 0; i < bytes; i++)
            o += (char)((v >> (8 * i)) & 0xFF);
    };
    for (const auto& [name, body] : files)
    {
        const uint32_t at = (uint32_t)z.size(), crc = crc32(body), sz = (uint32_t)body.size();
        le(z, 0x04034b50u, 4);
        le(z, 20, 2);
        le(z, 0, 2);
        le(z, 0, 2);
        le(z, 0, 2);
        le(z, 0x21, 2);
        le(z, crc, 4);
        le(z, sz, 4);
        le(z, sz, 4);
        le(z, (uint32_t)name.size(), 2);
        le(z, 0, 2);
        z += name;
        z += body;
        le(cd, 0x02014b50u, 4);
        le(cd, 20, 2);
        le(cd, 20, 2);
        le(cd, 0, 2);
        le(cd, 0, 2);
        le(cd, 0, 2);
        le(cd, 0x21, 2);
        le(cd, crc, 4);
        le(cd, sz, 4);
        le(cd, sz, 4);
        le(cd, (uint32_t)name.size(), 2);
        le(cd, 0, 2);
        le(cd, 0, 2);
        le(cd, 0, 2);
        le(cd, 0, 2);
        le(cd, 0, 4);
        le(cd, at, 4);
        cd += name;
    }
    const uint32_t cdAt = (uint32_t)z.size();
    z += cd;
    le(z, 0x06054b50u, 4);
    le(z, 0, 2);
    le(z, 0, 2);
    le(z, (uint32_t)files.size(), 2);
    le(z, (uint32_t)files.size(), 2);
    le(z, (uint32_t)cd.size(), 4);
    le(z, cdAt, 4);
    le(z, 0, 2);
    HANDLE fh = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    DWORD wrote = 0;
    const bool ok = fh != INVALID_HANDLE_VALUE && WriteFile(fh, z.data(), (DWORD)z.size(), &wrote, nullptr) &&
                    wrote == (DWORD)z.size();
    if (fh != INVALID_HANDLE_VALUE)
        CloseHandle(fh);
    if (!ok)
    {
        DeleteFileW(path.c_str());
        why = "the .pkl could not be written (" + wideToUtf8(path) + ")";
    }
    return ok;
}

// a --restore-model file's weights for the kernels: the .pkl the app keeps, or the author's ONNX
static bool ajNatLoad(const std::string& path, std::vector<uint8_t>& blob, std::vector<size_t>& off, int& nblocks,
                      std::string& why)
{
    AjWeights aw;
    const bool pkl = path.size() > 4 && _stricmp(path.c_str() + path.size() - 4, ".pkl") == 0;
    return (pkl ? ajFromPkl(utf8ToWide(path), aw, why) : ajFromOnnx(path, aw, why)) &&
           ajBlob(aw, blob, off, nblocks, why);
}

// --convert-restore-model IN.onnx OUT.pkl: AnimeJaNai's ONNX after the form check as the .pkl the app keeps, read
// back and checked to give the kernels the ONNX's block byte for byte; 0 = written, 2 = refused (the line says why)
static int ajConvert(const std::string& in, const std::wstring& out)
{
    AjWeights aw, back;
    std::vector<uint8_t> blob, blob2;
    std::vector<size_t> off, off2;
    int n = 0, n2 = 0;
    std::string why;
    if (!ajFromOnnx(in, aw, why) || !ajBlob(aw, blob, off, n, why) || !ajWritePkl(aw, out, why))
    {
        printf("convert: refused: %s\n", why.c_str());
        return 2;
    }
    if (!ajFromPkl(out, back, why) || !ajBlob(back, blob2, off2, n2, why) || blob != blob2 || off != off2)
    {
        DeleteFileW(out.c_str());
        printf("convert: refused: the .pkl does not read back as the ONNX's weights%s%s\n", why.empty() ? "" : ": ",
               why.c_str());
        return 2;
    }
    printf("convert: %d blocks, %zu tensors -> %s\n", n, (size_t)(2 * (3 + 4 * n)), wideToUtf8(out).c_str());
    return 0;
}

// a --restore-model file on the host's kernels: AnimeJaNai's form, else no Restore (read once a path, the line once)
static bool ajNatWanted()
{
    if (g_restoreModel.empty())
        return false;
    static std::string path;
    static bool ok = false;
    if (path != g_restoreModel)
    {
        path = g_restoreModel;
        std::vector<uint8_t> blob;
        std::vector<size_t> off;
        std::string why;
        int nb = 0;
        ok = ajNatLoad(path, blob, off, nb, why);
        if (ok)
            LOG("native: the Restore model runs on the host's kernels (SPAN-F3, %d blocks)\n", nb);
        else
            LOG("native: Restore unavailable (%s; a --restore-model runs only in AnimeJaNai HD's SPAN-F3 form, the "
                "Balanced model's)\n",
                why.c_str());
    }
    return ok;
}
