#pragma once

#include "../LineSearch.h"

#include "../Goldstein/Goldstein.h"

#include "../StrongWolfe/StrongWolfe.h"


namespace nlpp
{

template <class Function_>
struct DynamicLineSearch : public LineSearch<DynamicLineSearch<Function_>>
{
	using Base = LineSearch<DynamicLineSearch<Function_>>;
	using Function = wrap::LineSearch<wrap::FunctionGradient<Function_, fd::Gradient<Function_, fd::Forward>>, Vec>;


	DynamicLineSearch (std::string lsName = "Goldstein") : ls(std::make_unique<poly::Goldstein<Function>>())
	{
		if(lsName == "Goldstein")
			ls = std::make_unique<poly::Goldstein<Function>>();
		
		else if(lsName == "StrongWolfe")
			ls = std::make_unique<poly::StrongWolfe<Function>>();
	}
	
	DynamicLineSearch (const DynamicLineSearch& dls) : Base(dls), ls(dls.ls ? dls.ls->clone() : nullptr),
													   lineSearches(dls.lineSearches)
	{
	}


	DynamicLineSearch& operator= (const DynamicLineSearch& dls)
	{
		if(dls.ls)
			ls = std::unique_ptr<poly::LineSearch<Function>>(dls.ls->clone());
		
		lineSearches = dls.lineSearches;

		return *this;
	}

    
    double lineSearch (Function f)
    {
        return ls->lineSearch(f);
    }


	std::unique_ptr<poly::LineSearch<Function>> ls;
	
	std::vector<std::string> lineSearches;
};


} // namespace nlpp