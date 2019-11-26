/** @file
 *  @brief Some basic definitions and includes used by other files
*/
#pragma once

#include "helpers_dec.hpp"


namespace nlpp
{

namespace poly
{

template <class Impl>
std::unique_ptr<Impl> CloneBase<Impl>::clone () const
{
    return std::unique_ptr<Impl>(clone_impl());
}

template <class Base>
PolyClass<Base>::PolyClass(std::unique_ptr<Base> ptr) : impl(std::move(ptr))
{
}

template <class Base>
template <class Derived, std::enable_if_t<std::is_base_of_v<Base, Derived>, int>>
PolyClass<Base>::PolyClass (Derived&& derived) : impl(std::make_unique<std::decay_t<Derived>>(std::forward<Derived>(derived)))
{
}

template <class Base>
PolyClass<Base>::PolyClass (const PolyClass<Base>& polyClass) : impl(polyClass.impl ? polyClass.impl->clone() : nullptr)
{
}

template <class Base>
PolyClass<Base>& PolyClass<Base>::operator= (std::unique_ptr<Base> ptr)
{
    impl = std::move(ptr);
    return *this;
}

template <class Base>
template <class Derived>
PolyClass<Base>& PolyClass<Base>::operator= (Derived&& derived)
{
    if constexpr(std::is_base_of_v<Base, std::decay_t<Derived>>)
       impl = std::make_unique<std::decay_t<Derived>>(std::forward<Derived>(derived));

    else
        static_assert(false, "Wrong template parameter");

    return *this;
}


template <class Base>
PolyClass<Base>& PolyClass<Base>::operator= (const PolyClass<Base>& polyClass)
{
    if(polyClass.impl)
        impl = polyClass.impl->clone();

    return *this;
}

template <class Base>
Base* PolyClass<Base>::get () const
{
    return impl.get();
}

template <class Base>
Base* PolyClass<Base>::operator-> () const
{
    return get();
}

template <class Base>
template <class T>
void PolyClass<Base>::set (T&& t)
{
    operator=(std::forward<T>(t));
}


} // namespace poly

namespace impl
{

template <typename T, class V>
constexpr decltype(auto) cast (V&& v)
{
    return v.template cast<T>();
}

template <class V>
std::string toString (const V& x)
{
    std::stringstream ss;
    ss << x;
    return ss.str();
}

} // namespace impl

template <typename T>
inline constexpr decltype(auto) shift (T&& x)
{
    return std::forward<T>(x);
}

template <typename T, typename U, typename... Args>
inline constexpr decltype(auto) shift (T&& x, U&& y, Args&&... args)
{
    x = std::forward<U>(y);

    return shift(std::forward<U>(y), std::forward<Args>(args)...);
}

template <typename T>
inline constexpr int sign (T t)
{
    return int(T{0} < t) - int(t < T{0});
}

} // namespace nlpp