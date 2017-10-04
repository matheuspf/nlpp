#ifndef MODELO_H
#define MODELO_H

#include <bits/stdc++.h>
#include <assert.h>
#include <Eigen/Dense>

#include "Benchmark.h"

#define FORR(i, I, F) for(auto i = (I); i < (F); ++i)
#define RFORR(i, I, F) for(int i = (I); i >= (F); --i)
#define FOR(i, F) FORR(i, 0, F)
#define RFOR(i, F) RFORR(i, F, 0)
#define RG(v, I, F) begin(v) + (I), begin(v) + (F)
#define ALL(v) begin(v), end(v)
#define DB(...) cout << __VA_ARGS__ << '\n' << flush
#define DEBUG(...) cout << #__VA_ARGS__ << " = " << x << '\n'
#define pb push_back
#define F first
#define S second
#define ctx constexpr
#define fastio ios_base::sync_with_stdio(0); cin.tie(0);
#define unmap unordered_map
#define unset unordered_set
#define INF(T) numeric_limits<T>::max()
#define CIN(T) ({ T t; cin >> t; t; })
#define SIGN(x) (x > 0 ? 1 : x < 0 ? -1 : 0)


using namespace std;
using namespace Eigen;


using uchar = unsigned char;
using ll = long long;
using ull = unsigned long long;
using ld = long double;

using vi = vector<int>;
using vd = vector<double>;
using pd = pair<double, double>;


using Vec = VectorXd;
using Mat = MatrixXd;

using Veci = VectorXi;
using Mati = MatrixXi;

using Vecll = Matrix<long long, Dynamic, 1>;
using Matll = Matrix<long long, Dynamic, Dynamic>;


ctx double goldenRatio = 1.61803398875;


ctx double pi () { return 4 * atan(1); }


ctx double EPS = 1e-8;

ctx int cmpD (ld x, ld y, double e = EPS) { return (x <= y + e) ? (x + e < y) ? -1 : 0 : 1; }


ll mod (ll a, ll b) { return (a%b+b)%b; }

ll power (ll x, ll b)
{
	ll r = 1;

	for(; b; b >>= 1, x *= x)
		if(b & 1) r*= x;

	return r;
}


ostream& db () { return cout << '\n' << flush; }

template <typename T, typename... Args>
ostream& db (const T& t, const Args& ...args)
{
	cout << t;
	
	if(sizeof...(Args)) cout << ' ';

	return db(args...);
}


struct Rand
{
	Rand () : generator(random_device{}()) {}

	Rand (int seed) : generator(seed) {}

	mt19937 generator;
};


struct RandInt: Rand
{
	using Rand::Rand;

	inline int operator () (int min, int max) { return uniform_int_distribution<>(min, max - 1)(generator); }
};

struct RandDouble : Rand
{
	using Rand::Rand;
	
	inline double operator () (double min, double max) { return uniform_real_distribution<>(min, max)(generator); } 
};



namespace std
{
	template <typename T, size_t N>
	struct hash<array<T, N>>
	{
		inline size_t operator () (const array<T, N>& v) const
		{
			return accumulate(v.begin(), v.end(), size_t(0), [](T x, T y){ return x + hash<T>()(y); });
		}
	};


	template <typename T>
	struct hash<vector<T>>
	{
		inline size_t operator () (const vector<T>& v) const
		{
			return accumulate(v.begin(), v.end(), size_t(0), [](T x, T y){ return x + hash<T>()(y); });
		}
	};
}


template <typename T, size_t N>
decltype(auto) operator << (ostream& out, const array<T, N>& x)
{
	for(auto y : x) out << y << "  ";

	return out;
}

template <typename T>
decltype(auto) operator << (ostream& out, const vector<T>& x)
{
	for(auto y : x) out << y << "  ";

	return out;
}


template <class Container, typename T>
inline bool contains (Container&& c, T&& t)
{
	return c.find(t) != c.end();
}


template <typename T>
inline auto bound (const T& x, const T& l, const T& u)
{
	return std::min(u, std::max(l, x));
}



template <bool...>
struct And;

template <bool B1, bool... Bs>
struct And< B1, Bs... > : And< Bs... > {};

template <bool... Bs>
struct And< false, Bs... > : std::false_type {};

template <>
struct And<true> : std::true_type {};

template <>
struct And<> : std::true_type {};

template <bool... Bs>
ctx bool And_v = And< Bs... >::value;




namespace std
{
	template <class V, enable_if_t<is_same_v<decay_t<V>, Vec>, int> = 0>
	auto begin (V&& v)
	{
		return forward<V>(v).data();
	}

	template <class V, enable_if_t<is_same_v<decay_t<V>, Vec>, int> = 0>
	auto end (V&& v)
	{
		return forward<V>(v).data() + forward<V>(v).rows();
	}
}


double norm (double x) { return std::abs(x); }

double norm (const VectorXd& x) { return x.norm(); }




template <typename T>
ctx decltype(auto) shift (T&& x)
{
	return forward<T>(x);
}

template <typename T, typename U, typename... Args>
ctx decltype(auto) shift (T&& x, U&& y, Args&&... args)
{
	x = forward<U>(y);

	return shift(forward<U>(y), forward<Args>(args)...);
}



#endif // MODELO_H