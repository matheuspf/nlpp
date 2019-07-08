/** @file
 *  @brief Some basic definitions and includes used by other files
*/
#pragma once

#include "include/nlpp/Helpers/Helpers.h"


#define NLPP_USING_POLY_CLASS(ClassName, BaseName, ...) \
	using BaseName = __VA_ARGS__;	\
	using BaseName::BaseName;		\
	using BaseName::set;	 		\
	using BaseName::impl;			\
									\
	template <typename... Args>		\
	ClassName(Args&&...args) : BaseName(std::forward<Args>(args)...) {}


namespace nlpp::poly
{

template <class Impl>
struct CloneBase
{
    auto clone () const { return std::unique_ptr<Impl>(clone_impl()); }

    virtual Impl* clone_impl () const = 0;
};


template <class Base>
struct PolyClass
{
	virtual ~PolyClass () {}

    PolyClass (std::unique_ptr<Base> ptr) : impl(std::move(ptr))
	{
	}

	template <class Derived_, class Derived = std::decay_t<Derived_>,
		 	  std::enable_if_t<std::is_base_of<Base, Derived>::value, int> = 0>
	PolyClass (Derived_&& derived) : impl(std::make_unique<Derived>(std::forward<Derived>(derived)))
	{
	}

    PolyClass& operator= (std::unique_ptr<Base> ptr)
	{
		impl = std::move(ptr);
		return *this;
	}

	template <class Derived_, class Derived = std::decay_t<Derived_>,
		 	  std::enable_if_t<std::is_base_of<Base, Derived>::value, int> = 0>
	PolyClass& operator= (Derived_&& derived)
	{
		impl = std::make_unique<Derived>(std::forward<Derived>(derived));
		return *this;
	}

    PolyClass (const PolyClass& polyClass) : impl(polyClass.impl ? polyClass.impl->clone() : nullptr)
	{
	}

    PolyClass (PolyClass&& polyClass) = default;

    PolyClass& operator= (const PolyClass& polyClass)
	{
		if(polyClass.impl)
			impl = polyClass.impl->clone();
			
		return *this;
	}

    PolyClass& operator= (PolyClass&& polyClass) = default;


	Base* get () const
	{
		return impl.get();
	}

	Base* operator-> () const
	{
		return get();
	}


	template <class T>
	void set (T&& t)
	{
		operator=(std::forward<T>(t));
	}


    std::unique_ptr<Base> impl;
};

} // namespace nlpp::poly
