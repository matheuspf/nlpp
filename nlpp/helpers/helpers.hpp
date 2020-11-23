/** @file
 *  @brief Some basic definitions and includes used by other files
*/
#pragma once

#include "helpers_dec.hpp"


namespace nlpp
{

namespace impl
{

// Taken from https://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c
template <typename T>
std::string type_name()
{
    using TR = std::remove_reference_t<T>;
    std::unique_ptr<char, void(*)(void*)> own
           (
#ifndef _MSC_VER
                abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                           nullptr, nullptr),
#else
                nullptr,
#endif
                std::free
           );

    std::string r = own != nullptr ? own.get() : typeid(TR).name();

    if (std::is_const<TR>::value)
        r += " const";

    if (std::is_volatile<TR>::value)
        r += " volatile";

    if (std::is_lvalue_reference<T>::value)
        r += "&";

    else if (std::is_rvalue_reference<T>::value)
        r += "&&";

    return r;
}

} // namespace impl

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
template <class Derived, std::enable_if_t<std::is_base_of_v<Base, Derived>, int>>
PolyClass<Base>& PolyClass<Base>::operator= (Derived&& derived)
{
    impl = std::make_unique<std::decay_t<Derived>>(std::forward<Derived>(derived));
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

} // namespace nlpp