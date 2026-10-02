// Python's number rounding and formatting, reproduced exactly (the TypeScript render writes every
// stderr note and pass argument exactly as the python render did). JavaScript differs from python on all of them:
// Math.round and toFixed round a tie away from zero where python rounds half to even on the
// double's EXACT value, `f"{x:g}"` switches to exponent form at other thresholds than
// toPrecision, and `str(2.0)` is "2.0" where String(2.0) is "2". So the value is expanded to
// its exact decimal (a double is m * 2^e, exactly m * 5^-e / 10^-e) and rounded on that.

/** |x| = int / 10^scale, exactly (x finite). */
function exactDecimal(x: number): { int: bigint; scale: number } {
  const dv = new DataView(new ArrayBuffer(8));
  dv.setFloat64(0, Math.abs(x));
  const hi = dv.getUint32(0),
    lo = dv.getUint32(4);
  const expBits = (hi >>> 20) & 0x7ff;
  let mant = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let e: number;
  if (expBits === 0) e = -1074;
  else {
    mant |= 1n << 52n;
    e = expBits - 1075;
  }
  if (e >= 0) return { int: mant << BigInt(e), scale: 0 };
  return { int: mant * 5n ** BigInt(-e), scale: -e };
}

/** int / 10^scale re-expressed at `target` fractional digits, a tie rounded to even. */
function rescale(int: bigint, scale: number, target: number): bigint {
  if (target >= scale) return int * 10n ** BigInt(target - scale);
  const div = 10n ** BigInt(scale - target);
  let q = int / div;
  const r = int % div;
  const half = div / 2n; // div is a power of ten >= 10, so half is exact
  if (r > half || (r === half && (q & 1n) === 1n)) q += 1n;
  return q;
}

/** python round(x) for a float: the nearest integer, a tie to even. */
export function pyRound(x: number): number {
  const { int, scale } = exactDecimal(x);
  const q = Number(rescale(int, scale, 0));
  return x < 0 ? -q : q;
}

/** python format(x, `.{d}f`). */
export function pyFixed(x: number, d: number): string {
  const { int, scale } = exactDecimal(x);
  let s = rescale(int, scale, d).toString();
  if (d > 0) {
    s = s.padStart(d + 1, '0');
    s = s.slice(0, s.length - d) + '.' + s.slice(s.length - d);
  }
  return (x < 0 || Object.is(x, -0) ? '-' : '') + s;
}

/** python format(x, `.{p}g`) (`:g` = p 6). */
export function pyG(x: number, p = 6): string {
  const sign = x < 0 || Object.is(x, -0) ? '-' : '';
  if (x === 0) return sign + '0';
  if (!Number.isFinite(x)) return Number.isNaN(x) ? 'nan' : sign + 'inf';
  const { int, scale } = exactDecimal(x);
  const exp10 = int.toString().length - 1 - scale; // the leading digit's power of ten
  let sig = rescale(int, scale, p - 1 - exp10).toString();
  let X = exp10;
  if (sig.length > p) {
    X += 1;
    sig = sig.slice(0, p);
  } // 9.999995 -> 10.0000 carried a digit
  if (X < -4 || X >= p) {
    const m = (sig[0] + '.' + sig.slice(1)).replace(/0+$/, '').replace(/\.$/, '');
    return sign + m + 'e' + (X < 0 ? '-' : '+') + String(Math.abs(X)).padStart(2, '0');
  }
  let s: string;
  if (X >= 0) s = sig.slice(0, X + 1) + '.' + sig.slice(X + 1);
  else s = '0.' + '0'.repeat(-X - 1) + sig;
  if (s.includes('.')) s = s.replace(/0+$/, '').replace(/\.$/, '');
  return sign + s;
}

/** python repr(x) / str(x) for a float: the shortest round-trip digits, python's layout. */
export function pyFloatRepr(x: number): string {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? 'nan' : x < 0 ? '-inf' : 'inf';
  const sign = x < 0 || Object.is(x, -0) ? '-' : '';
  const [mant, expStr] = Math.abs(x).toExponential().split('e'); // shortest digits
  const digits = mant.replace('.', '');
  const E = Number(expStr);
  if (E < -4 || E >= 16) {
    const m = digits.length > 1 ? digits[0] + '.' + digits.slice(1) : digits;
    return sign + m + 'e' + (E < 0 ? '-' : '+') + String(Math.abs(E)).padStart(2, '0');
  }
  let s: string;
  if (E >= 0) {
    const ip = digits.slice(0, E + 1).padEnd(E + 1, '0');
    const fp = digits.slice(E + 1);
    s = ip + '.' + (fp || '0');
  } else s = '0.' + '0'.repeat(-E - 1) + digits;
  return sign + s;
}
