"""C++17 sources for the seven-qubit magic-cat lower-bound certificate.

The challenge runs new cert_*.py files in fork CI. The Python driver writes
these source texts to its temporary build directory and compiles four programs.
All intermediate tables are generated from the target on each invocation.
"""

KERNELS: dict[str, str] = {
    "exact.hpp": r'''// Gaussian-integer arithmetic.  No floating-point representation of a phase is used.
#pragma once
#include <limits>
#include <stdexcept>
#include <string>

// The certificate uses at most six logical qubits. Projecting a basis vector
// over an entire stabilizer group sums at most 64 fourth roots of unity.
struct Gaussian {
  int re, im;
  constexpr Gaussian(int r = 0, int i = 0) : re(r), im(i) {}
  constexpr int real() const { return re; }
  constexpr int imag() const { return im; }
  friend constexpr Gaussian operator+(Gaussian a, Gaussian b) {
    return {a.re + b.re, a.im + b.im};
  }
  friend constexpr Gaussian operator-(Gaussian a) { return {-a.re, -a.im}; }
  friend constexpr Gaussian operator-(Gaussian a, Gaussian b) { return a + (-b); }
  friend constexpr Gaussian operator*(Gaussian a, Gaussian b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
  }
  Gaussian &operator+=(Gaussian b) { return *this = *this + b; }
  friend constexpr bool operator==(Gaussian a, Gaussian b) {
    return a.re == b.re && a.im == b.im;
  }
  friend constexpr bool operator!=(Gaussian a, Gaussian b) { return !(a == b); }
};

inline int bit_count(unsigned int x) { return __builtin_popcount(x); }
inline int highest_bit(unsigned int x) {
  if (x == 0)
    throw std::logic_error("Pauli pivot requires a nonzero mask");
  return std::numeric_limits<unsigned int>::digits - 1 - __builtin_clz(x);
}
inline void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
''',
    "rank2.cpp": r'''#include <algorithm>
#include <array>
#include <chrono>
#include "exact.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;
using Z = Gaussian;
const Z im(0, 1);
const Z ii[] = {1, im, -1, -im};
// n and N describe the current logical state. Only the first N vector entries
// participate in a test.
int n = 0, N = 0;
using Vec = array<Z, 128>;
int pop(int x) { return bit_count(x); }
int par(int x) { return pop(x) & 1; }
int hi(int a) { return highest_bit(a); }
struct Basis {
  int b[14] = {}, d = 0;
  bool add(int a) {
    while (a && b[hi(a)])
      a ^= b[hi(a)];
    if (!a)
      return false;
    b[hi(a)] = a;
    d++;
    return true;
  }
};
struct Info {
  vector<int> stab;
};
int phase(Z a, Z b) {
  if (a == b)
    return 0;
  if (a == im * b)
    return 1;
  if (a == -b)
    return 2;
  if (a == -im * b)
    return 3;
  return -1;
}
// On an affine support a stabilizer phase is linear mod 4 plus twice a
// binary quadratic. Recover both from basis entries and test every point.
bool ray(const Vec &v, Info *info = nullptr) {
  int x0 = 0;
  while (x0 < N && v[x0] == Z(0, 0))
    x0++;
  if (x0 == N)
    return false;
  int cnt = 0, ph[128];
  vector<int> dirs;
  Basis D;
  for (int x = x0; x < N; x++)
    if (v[x] != Z(0, 0)) {
      int a = phase(v[x], v[x0]);
      if (a < 0)
        return false;
      ph[x] = a;
      cnt++;
      if (D.add(x ^ x0))
        dirs.push_back(x ^ x0);
    }
  if ((1 << D.d) != cnt)
    return false;
  int d = dirs.size(), Q[7][7] = {}, lin[7], points[128] = {}, vals[128] = {};
  for (int j = 0; j < d; j++) {
    int vj = dirs[j];
    lin[j] = ph[x0 ^ vj];
    for (int k = 0; k < j; k++) {
      int z = (4 + 4 + ph[x0 ^ vj ^ dirs[k]] - lin[j] - lin[k]) % 4;
      if (z & 1)
        return false;
      Q[j][k] = Q[k][j] = z / 2;
    }
    for (int m = 0; m < (1 << j); m++) {
      int add = 0;
      for (int k = 0; k < j; k++)
        if (m >> k & 1)
          add += Q[j][k];
      points[m + (1 << j)] = points[m] ^ vj;
      vals[m + (1 << j)] = (vals[m] + lin[j] + 2 * add) % 4;
      if (vals[m + (1 << j)] != ph[x0 ^ points[m + (1 << j)]])
        return false;
    }
  }
  if (info) {
    Basis rb;
    info->stab.clear();
    int signatures[128] = {};
    for (int z = 0; z < N; z++)
      for (int j = 0; j < d; j++)
        signatures[z] |= par(z & dirs[j]) << j;
    for (int z = 1; z < N; z++)
      if (!signatures[z] && rb.add(z << n))
        info->stab.push_back(z << n);
    for (int j = 0; j < d; j++) {
      int sig = (lin[j] % 2) << j;
      for (int k = 0; k < d; k++)
        if (k != j)
          sig |= Q[j][k] << k;
      int z = 0;
      while (z < N && signatures[z] != sig)
        ++z;
      require(z < N, "no Pauli generator for an affine phase function");
      info->stab.push_back(dirs[j] | (z << n));
    }
  }
  return true;
}
void pauli(const Vec &v, int P, Vec &w) {
  int xx = P & (N - 1), zz = P >> n;
  Z ph = ii[pop(xx & zz) & 3];
  for (int x = 0; x < N; x++)
    w[x ^ xx] = (par(x & zz) ? -ph : ph) * v[x];
}
// A Pauli killing one proposed summand leaves a stabilizer ray in one
// sector. The other sector is a ray, or differs from a ray by a Pauli
// translate of the survivor with relative phase in {1,i,-1,-i}.
bool rank2(const Vec &w) {
  if (ray(w))
    return true;
  for (int P = 1; P < N * N; P++) {
    Vec pw;
    pauli(w, P, pw);
    for (int sg : {1, -1}) {
      Vec v, y;
      bool zero = true;
      for (int x = 0; x < N; x++) {
        v[x] = w[x] + sg * pw[x];
        y[x] = w[x] - sg * pw[x];
        if (y[x] != Z(0, 0))
          zero = false;
      }
      if (zero)
        continue;
      Info f;
      if (!ray(v, &f))
        continue;
      if (ray(y))
        return true;
      Basis A;
      for (int b : f.stab)
        A.add(b);
      vector<int> errors = {0};
      for (int j = 0; j < 2 * n; j++)
        if (A.add(1 << j)) {
          int old = errors.size();
          for (int i = 0; i < old; i++)
            errors.push_back(errors[i] ^ (1 << j));
        }
      for (int Q : errors) {
        if (!par(((Q & (N - 1)) & (P >> n)) ^ ((P & (N - 1)) & (Q >> n))))
          continue;
        Vec q;
        pauli(v, Q, q);
        for (int ph = 0; ph < 4; ph++) {
          Vec res;
          for (int k = 0; k < N; k++)
            res[k] = y[k] - ii[ph] * q[k];
          if (ray(res))
            return true;
        }
      }
    }
  }
  return false;
}
''',
    "pairs.cc": r'''#include <array>
#include <algorithm>
#include "exact.hpp"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
using namespace std;
using U = uint64_t;
int pc(int x) { return bit_count(x); }
int hbit(int x) { return 1 << highest_bit(x); }
struct P {
  int x, z, s;
};
P mul(P a, P b) {
  int x = a.x ^ b.x, z = a.z ^ b.z;
  int ph = pc(a.x & a.z) + pc(b.x & b.z) - pc(x & z) + 2 * pc(a.z & b.x);
  require(!(ph & 1), "projector generators must commute");
  return {x, z, a.s * b.s * ((ph & 2) ? -1 : 1)};
}
P phy(P a) { return {a.x | (pc(a.x) % 2 << 6), a.z, a.s}; }
P lift(P b, P a) {
  int c = hbit(a.x ? a.x : a.z);
  int x = (b.x & (c - 1)) | ((b.x & ~(c - 1)) << 1),
      z = (b.z & (c - 1)) | ((b.z & ~(c - 1)) << 1);
  if (a.x)
    z |= c * ((pc(x & a.z) + pc(z & a.x)) % 2);
  else
    x |= c * (pc(x & a.z) % 2);
  return {x, z, b.s};
}
Gaussian ph[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
using C = Gaussian;
vector<C> half(const vector<C> &v, P a) {
  int n = v.size() / 2;
  vector<C> h(n);
  int c = hbit(a.x ? a.x : a.z);
  for (int j = 0; j < n; j++) {
    int k = (j & (c - 1)) | ((j & ~(c - 1)) << 1);
    if (!a.x) {
      if ((pc(k & a.z) % 2) != (a.s < 0))
        k ^= c;
      h[j] = v[k];
    } else
      h[j] = v[k] + a.s * (pc((k ^ a.x) & a.z) % 2 ? -1 : 1) *
                        ph[pc(a.x & a.z) % 4] * v[k ^ a.x];
  }
  return h;
}
int fixspan[256][8];
// Reduce the local ordered Pauli pair under the (X+Y)/sqrt(2) action.
// A histogram quotients by all qubit permutations. The three low bits
// track gate parity and both signed generators; fixspan removes the
// freedom to act on sites whose two Pauli letters stay fixed.
U key(P a, P b) {
  U hist = 0;
  int mods = 0, u = 2 * (a.s < 0) + (b.s < 0);
  for (int j = 0; j < 7; j++) {
    int x = ((a.x >> j) & 1) * 2 + ((a.z >> j) & 1),
        y = ((b.x >> j) & 1) * 2 + ((b.z >> j) & 1);
    int z = 4 * x + y, alt = 4 * (x ^ (x >= 2)) + (y ^ (y >= 2)),
        v = 4 + 2 * (x == 1) + (y == 1);
    if (z == alt)
      mods |= 1 << v;
    else if (z > alt) {
      u ^= v;
      z = alt;
    }
    hist += U(1) << (4 * z);
  }
  return hist * 8 + fixspan[mods][u]; // canonical symbols are at most 11, so
                                      // the packed histogram fits
}
// Any ordered pair of independent generators of the same two-check
// projector, and either physical representative modulo even parity, works.
U canon(P a, P b) {
  U ret = ~U(0);
  P Z{0, 127, 1};
  P pp[3] = {a, b, mul(a, b)};
  for (int c = 0; c < 3; c++)
    for (int d = 0; d < 3; d++)
      if (c != d)
        for (int u = 0; u < 4; u++)
          ret = min(ret, key(u & 1 ? mul(pp[c], Z) : pp[c],
                             u & 2 ? mul(pp[d], Z) : pp[d]));
  return ret;
}
int main(int argc, char **argv) {
  require(argc == 2, "usage: pairs OUTPUT_DIRECTORY");
  string d = argv[1];
  for (int m = 0; m < 256; m++) {
    bool span[8] = {1};
    for (int i = 0; i < 8; i++)
      if (m >> i & 1)
        for (int j = 0; j < 8; j++)
          if (span[j])
            span[j ^ i] = 1;
    for (int j = 0; j < 8; j++) {
      fixspan[m][j] = 8;
      for (int k = 0; k < 8; k++)
        if (span[k])
          fixspan[m][j] = min(fixspan[m][j], j ^ k);
    }
  }
  const int reps[] = {-31, -7, -1, 1, 7, 31, 64, 66, 78, 192, 196, 448};
  vector<C> C7(64);
  for (int i = 0; i < 64; i++)
    C7[i] = ph[((pc(i) + 1) / 2) % 4];
  unordered_map<U, int> ids;
  ofstream idsfile(d + "/second_ids.txt"), states(d + "/second_states.dat"),
      first(d + "/probe_states.dat");
  ofstream tar(d + "/C7.dat");
  for (auto stream : {&idsfile, &states, &first, &tar})
    stream->exceptions(ios::badbit | ios::failbit);
  tar << 1;
  for (C z : C7)
    tar << " " << z.real() << " " << z.imag();
  tar << "\n";
  for (int r : reps) {
    int a = abs(r);
    P p{a % 64, a / 64, r > 0 ? 1 : -1};
    vector<C> w = half(C7, p);
    first << r;
    for (C z : w)
      first << " " << z.real() << " " << z.imag();
    first << "\n";
    for (int j = 1; j < 1024; j++)
      for (int s : {1, -1}) {
        P q{j % 32, j / 32, s};
        U k = canon(phy(p), phy(lift(q, p)));
        int id;
        if (!ids.count(k)) {
          id = ids.size();
          ids[k] = id;
          vector<C> v = half(w, q);
          states << id;
          for (auto z : v)
            states << " " << z.real() << " " << z.imag();
          states << "\n";
        } else
          id = ids[k];
        idsfile << r << " " << s * j << " " << id << "\n";
      }
  }
  for (auto stream : {&idsfile, &states, &first, &tar})
    stream->close();
  cerr << "unique types " << ids.size() << endl;
}
''',
    "third.cc": r'''#include "rank2.cpp"
// Decode a signed Pauli projection by deleting its pivot coordinate.
void half(const Vec &w, int P, int sg, Vec &dst) {
  int xx = P & (N - 1), zz = P >> n;
  int b = xx ? hi(xx) : hi(zz);
  int c = 1 << b;
  Z f = ii[pop(xx & zz) % 4] * sg;
  for (int j = 0; j < N / 2; j++) {
    int x = (j & (c - 1)) + ((j & ~(c - 1)) << 1);
    if (!xx) {
      if ((par(x & zz) ? -1 : 1) != sg)
        x ^= c;
      dst[j] = w[x];
    } else {
      dst[j] = w[x] + (par((x ^ xx) & zz) ? -f : f) * w[x ^ xx];
    }
  }
}
int main(int argc, char **argv) {
  require(argc == 3, "usage: third N SECOND_STATES");
  n = stoi(argv[1]);
  require(n == 4, "expected four-qubit second projected states");
  N = 1 << n;
  ifstream fin(argv[2]);
  require(fin.is_open(), "missing second-projection states");
  int id;
  while (fin >> id) {
    Vec v = {};
    for (int x = 0; x < N; x++) {
      int a, b;
      require(bool(fin >> a >> b), "truncated second-projection state");
      v[x] = Z(a, b);
    }
    int exact = 0, num = 0;
    vector<int> valid;
    auto start = chrono::steady_clock::now();
    for (int P = 1; P < N * N; P++) {
      bool signpass = false;
      for (int s : {1, -1}) {
        Vec h = {};
        half(v, P, s, h);
        n--;
        N /= 2;
        bool zero = true;
        for (int i = 0; i < N; i++)
          if (h[i] != Z(0, 0))
            zero = false;
        bool b = zero || rank2(h);
        N *= 2;
        n++;
        if (b) {
          exact++;
          signpass = true;
          valid.push_back(P * s);
        }
      }
      num += signpass;
    }
    cout
        << id << " " << num << " 0 " << exact << " "
        << chrono::duration<double>(chrono::steady_clock::now() - start).count()
        << " |";
    for (int a : valid)
      cout << " " << a;
    cout << endl;
  }
  require(fin.eof(), "invalid second-projection state identifier");
}
''',
    "lagrangians.cc": r'''#include <algorithm>
#include <array>
#include "exact.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;
using C = Gaussian;
const C ii[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
int n, N, limit;
vector<int> prod;
vector<char> allowed;
vector<vector<int>> rays;
int pc(int x) { return bit_count(x); }
int mul(int x, int y) {
  int a = abs(x) - 1, b = abs(y) - 1, c = a ^ b,
      ph = prod[a] + prod[b] - prod[c] + 2 * pc((a >> n) & b);
  if (ph & 1)
    return 0;
  return (c + 1) * ((ph & 2) ? -1 : 1) * (x < 0 ? -1 : 1) * (y < 0 ? -1 : 1);
}
int key(int a) { return a + N * N; }
// L is the generated signed subgroup and G its independent generators.
// The next generator is the least unsigned label in its coset of L, with
// strictly increasing leading bit. These echelon rules give every subgroup
// exactly once. Each newly generated group element must be an allowed check.
void visit(vector<int> &L, vector<int> &w, vector<int> &G) {
  if ((int)rays.size() >= limit)
    return;
  if ((int)L.size() == N) {
    rays.push_back(G);
    return;
  }
  if ((w.size() + 1) * L.size() < static_cast<size_t>(N))
    return;
  for (size_t i = 0; i < w.size() && (int)rays.size() < limit; i++) {
    int p = w[i], v = abs(p) - 1, old = L.size();
    bool ok = true;
    for (int j = 0; j < old; j++) {
      int t = mul(p, L[j]);
      if (!t || !allowed[key(t)]) {
        ok = false;
        break;
      }
      L.push_back(t);
    }
    if (ok) {
      G.push_back(p);
      vector<int> out;
      for (size_t j = i + 1; j < w.size(); j++) {
        int q = w[j], z = mul(q, p);
        if (highest_bit(abs(q) - 1) <= highest_bit(v) || !z ||
            !allowed[key(z)])
          continue;
        bool good = true;
        for (size_t k = 1; k < L.size(); k++) {
          int c = mul(L[k], q);
          if (!c || abs(c) <= abs(q)) {
            good = false;
            break;
          }
        }
        if (good)
          out.push_back(q);
      }
      visit(L, out, G);
      G.pop_back();
    }
    L.resize(old);
  }
}
vector<C> build(const vector<int> &G) {
  vector<C> v(N);
  for (int t = 0; t < N; t++) {
    fill(v.begin(), v.end(), C{});
    v[t] = 1;
    for (int a : G) {
      int p = abs(a) - 1, x = p & (N - 1), z = p >> n, s = a < 0 ? -1 : 1;
      vector<C> old = v;
      for (int j = 0; j < N; j++)
        v[j ^ x] += old[j] * s * (pc(j & z) % 2 ? -1 : 1) * ii[prod[p]];
    }
    bool ok = false;
    for (auto a : v)
      if (a.real() || a.imag())
        ok = true;
    if (ok)
      return v;
  }
  throw logic_error("no ray");
}
// 256 is a square root of -1 modulo 65537. After projective scaling,
// stabilizer coordinates are zero or fourth roots of unity. For up to
// five such columns Hadamard bounds the squared norm of any square minor
// by 5^5 < P. A nonzero independent minor therefore survives reduction.
constexpr int P = 65537;
static_assert((256 * 256 + 1) % P == 0 && 5 * 5 * 5 * 5 * 5 < P);
int inv[P];
int mod(long long x) {
  x %= P;
  return x < 0 ? x + P : x;
}
using V = vector<int>;
V encode(const vector<C> &v) {
  V a;
  for (auto x : v)
    a.push_back(mod(x.real() + 256 * x.imag()));
  return a;
}
int pivot(V &v) {
  for (int i = 0; i < static_cast<int>(v.size()); i++)
    if (v[i])
      return i;
  return -1;
}
void elim(V &v, const V &u, int p) {
  int s = v[p];
  if (s)
    for (int i = p; i < N; i++)
      v[i] = mod(v[i] - (long long)s * u[i]);
}
bool norm(V &v, int p) {
  if (p < 0)
    return false;
  int s = inv[v[p]];
  for (int j = p; j < N; j++)
    v[j] = mod((long long)v[j] * s);
  return true;
}

struct TCol {
  V v;
  int sig, p;
};
bool subcol(TCol &c, const TCol &b) {
  int k = c.v[b.p];
  if (k) {
    elim(c.v, b.v, b.p);
    c.sig = mod(c.sig - (long long)k * b.sig);
  }
  c.p = pivot(c.v);
  if (c.p < 0)
    return c.sig;
  int f = inv[c.v[c.p]];
  c.sig = mod((long long)c.sig * f);
  norm(c.v, c.p);
  return false;
}
// Quotient columns by the target and successively chosen rays. sig retains
// the target coefficient. Equal projective columns with different sig then
// supply a span containing the target.
bool go(const vector<TCol> &cs, int lev) {
  if (lev < 2)
    return false;
  if (lev == 2) {
    unordered_map<string, int> seen;
    for (auto &c : cs) {
      string key((char *)c.v.data(), N * sizeof(int));
      auto q = seen.emplace(key, c.sig);
      if (!q.second && q.first->second != c.sig)
        return true;
    }
    return false;
  }
  for (size_t i = 0; i < cs.size(); i++) {
    vector<TCol> ls;
    for (size_t j = i + 1; j < cs.size(); j++) {
      TCol c = cs[j];
      if (subcol(c, cs[i]))
        return true;
      if (c.p >= 0)
        ls.push_back(move(c));
    }
    if (go(ls, lev - 1))
      return true;
  }
  return false;
}
bool rankk(const vector<C> &tar, int k) {
  V t = encode(tar);
  int p = pivot(t);
  if (p < 0)
    return true;
  norm(t, p);
  TCol b{t, -1, p};
  vector<TCol> ls;
  for (auto &g : rays) {
    TCol c{encode(build(g)), 0, 0};
    if (subcol(c, b))
      return true;
    if (c.p >= 0)
      ls.push_back(move(c));
  }
  return go(ls, k);
}
int main(int argc, char **argv) {
  require(argc == 6 || argc == 7, "usage: lagrangians N LIMIT CHECKS TARGETS RANK [RAYS]");
  n = stoi(argv[1]);
  require(n >= 4 && n <= 6, "expected four to six logical qubits");
  N = 1 << n;
  limit = stoi(argv[2]);
  require(limit > 0, "positive candidate limit required");
  prod.resize(N * N);
  allowed.resize(2 * N * N + 1);
  for (int p = 0; p < N * N; p++)
    prod[p] = pc((p & (N - 1)) & (p >> n)) % 4;
  inv[1] = 1;
  for (int i = 2; i < P; i++)
    inv[i] = P - mod((long long)(P / i) * inv[P % i]);
  map<int, vector<C>> targets;
  {
    ifstream tf(argv[4]);
    require(tf.is_open(), "missing target states");
    int k;
    while (tf >> k) {
      vector<C> v(N);
      for (auto &z : v) {
        int a, b;
        require(bool(tf >> a >> b), "truncated target state");
        z = {a, b};
      }
      require(targets.emplace(k, v).second, "duplicate target id");
    }
    require(tf.eof(), "invalid target id");
  }
  ifstream f(argv[3]);
  require(f.is_open(), "missing allowed-check file");
  string l;
  while (getline(f, l)) {
    stringstream id(l);
    int r;
    require(bool(id >> r), "missing row identifier in allowed-check list");
    size_t sep = l.find('|');
    require(sep != string::npos, "invalid allowed-check row");
    stringstream v(l.substr(sep + 1));
    fill(allowed.begin(), allowed.end(), 0);
    vector<int> w;
    int h;
    while (v >> h) {
      require(h > -N * N && h < N * N && h != 0, "invalid signed check");
      int p = (abs(h) + 1) * (h > 0 ? -1 : 1);
      require(!allowed[key(p)], "duplicate signed check");
      allowed[key(p)] = 1;
      w.push_back(p);
    }
    require(v.eof(), "invalid signed-check list");
    sort(w.begin(), w.end(), [](int a, int b) { return abs(a) < abs(b); });
    rays.clear();
    vector<int> L{1}, G;
    visit(L, w, G);
    if (argc > 6) {
      ofstream gg(argv[6]);
      gg.exceptions(ios::badbit | ios::failbit);
      for (auto &g : rays) {
        for (int p : g)
          gg << p << " ";
        gg << "|";
        for (C z : build(g))
          gg << " " << z.real() << " " << z.imag();
        gg << "\n";
      }
      gg.close();
    }
    // A truncated enumeration is always retained as possibly low rank.
    require(targets.count(r), "allowed-check row has no target");
    bool possible = static_cast<int>(rays.size()) >= limit ||
                    rankk(targets.at(r), stoi(argv[5]));
    cout << r << " " << rays.size() << " " << possible << endl;
  }
  require(f.eof(), "failed reading allowed checks");
}
''',
    "five_span.cc": r'''#include <algorithm>
#include <array>
#include "exact.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
constexpr int N = 64, R = 1088, P = 65537;
static_assert((256 * 256 + 1) % P == 0 && 5 * 5 * 5 * 5 * 5 < P);
int iv[P], gen[12][R], b2[R], b3[R];
struct C {
  array<int, N> v;
  int sig, p;
};
C t, cols[R];
int mod(long long x) {
  x %= P;
  return x < 0 ? x + P : x;
}
int fix(C &a) {
  int p = 0;
  while (p < N && !a.v[p])
    p++;
  a.p = p;
  if (p == N)
    return p;
  int z = iv[a.v[p]];
  for (int i = p; i < N; i++)
    a.v[i] = mod((long long)z * a.v[i]);
  a.sig = mod((long long)z * a.sig);
  return p;
}
void sub(C &a, const C &b) {
  int f = a.v[b.p];
  if (f) {
    for (int i = b.p; i < N; i++)
      a.v[i] = mod(a.v[i] - (long long)f * b.v[i]);
    a.sig = mod(a.sig - (long long)f * b.sig);
  }
}
uint64_t pack(int a, int b, int c) {
  return a | (uint64_t(b) << 11) | (uint64_t(c) << 22);
}
int main(int argc, char **argv) {
  require(argc == 2, "usage: five_span OUTPUT_DIRECTORY");
  string base = argv[1];
  iv[1] = 1;
  for (int i = 2; i < P; i++)
    iv[i] = P - mod((long long)(P / i) * iv[P % i]);
  ifstream target(base + "/C7.dat");
  int id;
  require(bool(target >> id) && id == 1, "invalid target id");
  t.sig = P - 1;
  for (auto &v : t.v) {
    int a, b;
    require(bool(target >> a >> b), "truncated final target");
    v = mod(a + 256 * b);
  }
  require(fix(t) != N, "target vanished modulo the prime");
  ifstream ff(base + "/candidates.txt");
  require(ff.is_open(), "missing candidates");
  string l;
  int k = 0;
  while (getline(ff, l)) {
    auto separator = l.find('|');
    require(k < R && separator != string::npos, "invalid candidate or too many rays");
    stringstream f(l.substr(separator + 1));
    auto &a = cols[k++];
    for (auto &v : a.v) {
      int x, y;
      require(bool(f >> x >> y), "truncated candidate ray");
      v = mod(x + 256 * y);
    }
    sub(a, t);
    if (fix(a) == N && a.sig) {
      cout << "rank 1" << endl;
      return 1;
    }
  }
  require(ff.eof() && k == R, "incorrect number of candidate rays");
  ifstream gf(base + "/orbit_perms.txt");
  require(gf.is_open(), "missing ray permutations");
  for (auto &g : gen) {
    array<bool, R> seen = {};
    for (auto &v : g) {
      require(bool(gf >> v) && v >= 0 && v < R && !seen[v], "invalid ray permutation");
      seen[v] = true;
    }
  }
  for (int i = 0; i < R; i++) {
    b2[i] = i * (i - 1) / 2;
    b3[i] = i * (i - 1) * (i - 2) / 6;
  }
  // An orbit of unordered triples is explored once. After quotienting by
  // three pivots, the remaining two rays need only a projective-key lookup.
  vector<bool> visited(size_t(R) * (R - 1) * (R - 2) / 6);
  vector<uint64_t> q;
  int ct = 0, exam = 0;
  for (int c = 2; c < R; c++)
    for (int b = 1; b < c; b++)
      for (int a = 0; a < b; a++) {
        int id = a + b2[b] + b3[c];
        if (visited[id])
          continue;
        ct++;
        q.assign(1, pack(a, b, c));
        visited[id] = 1;
        for (size_t j = 0; j < q.size(); j++) {
          int x = q[j] & 2047, y = (q[j] >> 11) & 2047, z = q[j] >> 22;
          for (auto &g : gen) {
            int A = g[x], B = g[y], C = g[z];
            if (A > B)
              swap(A, B);
            if (B > C)
              swap(B, C);
            if (A > B)
              swap(A, B);
            int s = A + b2[B] + b3[C];
            if (!visited[s]) {
              visited[s] = 1;
              q.push_back(pack(A, B, C));
            }
          }
        }
        vector<C> piv;
        for (int j : {a, b, c}) {
          C v = cols[j];
          for (auto &w : piv)
            sub(v, w);
          if (fix(v) == N) {
            if (v.sig) {
              cout << "rank<=3 " << a << " " << b << " " << c << endl;
              return 1;
            }
          } else
            piv.push_back(v);
        }
        if (piv.size() < 3)
          continue;
        exam++;
        unordered_map<string, pair<int, int>> seen;
        for (int j = 0; j < R; j++)
          if (j != a && j != b && j != c) {
            C v = cols[j];
            for (auto &w : piv)
              sub(v, w);
            if (fix(v) == N) {
              if (v.sig) {
                cout << "rank<=4 " << a << " " << b << " " << c << " " << j
                     << endl;
                return 1;
              }
              continue;
            }
            string key((char *)v.v.data(), N * sizeof(int));
            auto it = seen.emplace(key, make_pair(v.sig, j));
            if (!it.second && it.first->second.first != v.sig) {
              cout << "rank<=5 " << a << " " << b << " " << c << " "
                   << it.first->second.second << " " << j << endl;
              return 1;
            }
          }
      }
  cout << "unordered triples " << visited.size() << " orbits " << ct
       << " independent modulo target " << exam << ", no rank-five span"
       << endl;
}
''',
}
