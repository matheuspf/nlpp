/** \file MDE.h
  *
  * This is the main file. Contains the implementation of the 'MDE' (stands 
  * for Modified Differential Evolution) metaheuristic described in:
  *
  *  http://ieeexplore.ieee.org/document/1688286/
  *
  * It was one of the competing methods of IEEE Congress on Evolutionary
  * Computation of 2006, on the Special Session on Constrained Real-Parameter 
  * Optimization. Here, I will only present brief comments about the method.
  * For any detailed information, please, refer to the original paper.
*/

#ifndef DE_DPS_DE_DPS_H
#define DE_DPS_DE_DPS_H

#include <type_traits>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <ctime>
#include <chrono>
#include <map>
#include <assert.h>
#include <functional>
#include <iostream>

#include "Random.h"
#include "Function.h"

#include <iostream>



namespace mde
{
    /** These are the parameters of the MDE algorithm. They are created on a separate
      * class, so it is much easier to the user to define these parameters first and
      * then create a 'MDE' class, passing them to the constructor. I comment very 
      * briefly about them. For detailed information, please, refer to the paper.
    */
    struct Parameters
    {
        Parameters(int popSize = 30, double Fa = 0.8, double Fb = 0.1, 
                   double Cr = 0.9, double Srmax = 0.55, double Srmin = 0.025, 
                   int children = 5, int maxIter = 3333, double eqTol = 1e-10, int debug = 0,
                   std::string bndHandle = "conservate") : popSize(popSize), Fa(Fa), Fb(Fb), Cr(Cr), 
                                                           Srmax(Srmax), Srmin(Srmin), Sr(Srmax),
                                                           children(children), maxIter(maxIter), 
                                                           eqTol(eqTol), bndHandle(bndHandle), debug(debug) {}


        int N;        /// The number of elements in each vector. Defined by the user function
        int popSize;  /// The number of elements in the population

        double Fa;    /// First factor for modified differential mutation
        double Fb;    /// Second factor for modified differential mutation
        double Cr;    /// Pobability factor for modified differential mutation

        double Srmax; /// Initial factor that controls the probability of accepting a solution with a smaller fitness only
        double Srmin; /// The same as above, but this is a lower bound
        double Sr;    /// The current value of this factor

        int children; /// Number of different children generated in the modified differential mutation
        int maxIter;   /// Maximum iterations of the MDE

        double eqTol;  /// The tolerance factor for the equality constraints. See the 'SetValues' class

        int debug;


        /** this is an option to choose how solutions generated outside the bounds should
         *  be treated. I implemented some methods described here:
         *
         *  https://elektron.elka.pw.edu.pl/~jarabas/ALHE/krakow1.pdf
         *
         *  These are the available methods: "clip" (default), "conservate", "conservateall", 
         *  "reinitialize". I comment very briefly how they work above their definition. 
         *  Again, for more details, please, refer to the papers.
        */
        std::string bndHandle;
    };





    /** The main class. It only has a few parameters and type definitions. If you want
      * (it is optional) to change the default value for some parameter, here is the way:
      *
      * mde::Parameters params;      // Create a 'mde::Parameters' class
      *
      * params.popSize = 100;                  // Change the population size
      * params.bndHandle = "conservate";       // Change the bound handling function
      *
      * mde::MDE<FunctionType> myMde(params);    // Create a 'mde::MDE' class with 'FunctionType' function passing 
                                                 // the 'params' as arguments. All the other arguments are default
      * auto best = myMde();     // Execute and retrieve the best element                           
    */

    template <class FunctionType>
    class MDE : Parameters
    {
    public:

        using Function   = SetValues<FunctionType>;    /// The 'Function' type is inherited by 'SetValues'
        using Vector     = typename Function::Vector;  /// The 'Vector' type is defined by 'FunctionType'
        using Population = std::vector<Vector>;        /// The 'Population' type is a 'std::vector' of 'Vector's


        /// Type of the function pointer for the bounds handling functions. See the 'Parameters' class
        using BoundsHandle = void (MDE::*)(Vector&, const Vector&);


        /// Here you can pass a 'Parameters' class and an 'FunctionType' with the parameters you want
        MDE (const Parameters& param,
             const FunctionType& function = FunctionType()) : Parameters(param), permutation(popSize), 
                                                              population(popSize), function(eqTol, function)
        {
            initialize();  /// Call the initialization function
        }


        /// Here the parameters are all default, and you can pass an 'FunctionType' with the parameters you want
        MDE (const FunctionType& function = FunctionType()) : permutation(popSize), population(popSize), 
                                                              function(eqTol, function)
        {
            initialize();  /// Call the initialization function
        }



        /// Main function. Execute the MDE algorithm
        Vector operator () ()
        {
            /// Sort the population according to the comparison function defined in the 'mde::Vector' class
            std::sort(population.begin(), population.end());

            best = population.front();    /// The best element is always at the first position

            int iter = 0;    /// Iteration counter


            /// Outter loop. Checks convergence and maximum iterations
            while(!converged(best) && iter++ < maxIter)
            {
                /// First inner loop. Iterates through all elements of the population
                for(int i = 0; i < population.size(); ++i)
                {
                    Vector& parent = population[i];     /// The current parent

                    Vector bestChild(N);   /// The best child from all the generated children for the current parent

                    /// Generate 'children' 
                    for(int k = 0; k < children; ++k)
                    {
                        /** This is the easiest way (although not the fastest) to unbiasedly generate
                          * three different random numbers that also differ from 'i'. These numbers are
                          * the indexes for the three vectors needed for the modified differential mutation
                        */ 
                        std::random_shuffle(permutation.begin(), permutation.end());

                        const Vector& x1 = population[permutation[0] != i ? permutation[0] : permutation.back()];
                        const Vector& x2 = population[permutation[1] != i ? permutation[1] : permutation.back()];
                        const Vector& x3 = population[permutation[2] != i ? permutation[2] : permutation.back()];

                        /// Perform the modified differential mutation and return a child
                        Vector child = differentialMutation(x1, x2, x3, parent);

                        /// Calling a member function with 'std::function' is easy like that
                        boundsHandle(this, child, parent);


                        function(child);    /// Set fitness and violation for the new vector

                        bestChild = std::min(bestChild, child);   /// Take the best between both
                    }

                    /// If convergence is reached, set 'best' to 'bestChild' and return it
                    if(converged(bestChild))
                        return (best = bestChild);


                    /** This is a feature of the 'MDE' method. With probability 'Sr' (that is 
                      * calculated based on the values of 'Srmax' and 'Srmin') we compare the
                      * vectors based only on their fitness values, instead of making the MDE
                      * comparison. This is helpfull in situations where a solution is slightly
                      * infeasible, but may be close to a optimum. In earlier stages of the
                      * search, 'Sr' assumes greater values starting from 'Srmax', and then
                      * decreases at every iteration, reaching 'Srmin'.
                    */
                    if(randDouble(0.0, 1.0) < Sr)
                    {
                        if(bestChild.fitness < parent.fitness) /// Compare only the fitness value and take the best
                            parent = bestChild;
                    }

                    else
                        parent = std::min(parent, bestChild);  /// Use MDE comparison and thake the best

                    best = std::min(best, bestChild);   /// Take the best between both (using MDE comparison)
                }

                std::sort( population.begin(), population.end() );  /// Sort MDE population

                /** The formula for calculating the 'Sr' probability. It drecreases smoothly in
                  * the first (maxIter / 3) iterations. Then, it is set permanently to 'Srmin'.
                */
                Sr = (iter < (maxIter / 3) ? Sr - (3.0 / maxIter) * (Srmax - Srmin) : Srmin);



                if(debug)
                {
                	std::cout << "Iteration: " << iter << "\n";
                	std::cout << "Best individual: [";

                    for(int i = 0; i < best.size(); ++i)
                    {
                        if(i)
                            std::cout << ", ";

                        std::cout << best[i];
                    }

                	std::cout << "]\n";

                	std::cout << "Best Fitness: " << best.fitness << "\n";
                	std::cout << "Best Violation: " << best.violation << "\n\n\n";
                }
            }

            /// Return best found solution (not the optimal one if 'function.optimal' is specified)
            return best;
        }


        /// Initialization procedure. Must be called if you want to run the algorithm again from scratch
        void initialize ()
        {
            /// A mapping from a string to a pointer to member function for handling the bounds
            static std::map<std::string, BoundsHandle> bndMap = {{ "conservate",    &MDE::conservate   },
                                                                 { "clip",          &MDE::clip         },
                                                                 { "reinitialize",  &MDE::reinitialize }};

            /// Make the all the input lower case                                        
            std::transform(bndHandle.begin(), bndHandle.end(), bndHandle.begin(), ::tolower);


            /// Verify if the option exists and map to it
            auto it = bndMap.find(bndHandle);

            assert(it != bndMap.end() && "Invalid Bound handling option");

            boundsHandle = it->second;



            /** Here we deal with unnitialized number of variables or lower and upper bounds.
              * If 'N' was not initialized (neither by compile or runtime values), we set it to 
              * the max of the size of the lower and upper bounds (well, why in the heck would you
              * provide different non zero lenghts for lower and upper bounds anyways???). If neither
              * lower or upper bounds are initialized, we stop here. If either the lower or the upper
              * bound vectors are not initialized, they are set to a 'N' dimensional vector with
              * -1e8 (lower bound) or 1e8 (upper bound).
            */
            N = function.N;

            if(N == 0)
                N = function.N = std::max(function.upperBounds.size(), function.lowerBounds.size());

            assert(N && "Zero variables???");

            if(function.lowerBounds.empty())
                function.lowerBounds = Vector(N, -1e8);

            if(function.upperBounds.empty())
                function.upperBounds = Vector(N, 1e8);


            /// Initializes the permutation vector used for selecting the three different vectors for the mutation
            std::iota(permutation.begin(), permutation.end(), 0);
    


            /// Initializes a random population
            for(auto& x : population)
            {
                x = newVector();  /// 'N' dimensional 'Vector' class

                function(x);  /// Calculate both fitness and violation for vector 'x'
            }
        }



        /// Returns a new 'N' dimensional 'Vector' class uniformly distributed within the box
        Vector newVector ()
        {
            Vector x(N);    /// Dummy constructor if it is a 'std::array'

            /// Random value for each dimension
            for(int i = 0; i < N; ++i)
                x[i] = randDouble(function.lowerBounds[i], function.upperBounds[i]);

            return x;
        }



        /// Modified differential mutation
        Vector differentialMutation (const Vector& x1, const Vector& x2, const Vector& x3, const Vector& parent)
        {
            Vector child(N);   /// The resulting child

            int jRand = randInt(0, N);   /// This component is guaranteed to not get a value from the parent


            /** It works as follows. With probability 'Cr' or if 'j' == 'jRand', we set the component 'j' 
              * of the child to the modified differential mutation, which is a weighted sum of 'x3[j]', 
              * a factor 'Fa' times the difference between 'best[j]' and 'x2[j]' and a factor 'Fb' times 
              * the  difference between 'parent[j]' and 'x1'. 'Fa' is set to a large value (0.8) while 
              * 'Fb' assumes a small value (0.1). So, the contribution of the best found element is
              * greater than the contribution of the parent, increasing the selective pressure.
            */
            for (int j = 0; j < N; ++j)
            {
                if (randDouble(0, 1.0) < Cr || j == jRand)
                    child[j] = x3[j] + Fa * (best[j] - x2[j]) + Fb * (parent[j] - x1[j]);

                else
                    child[j] = parent[j];
            }

            return child;
        }



        /// Checks if the value 'x' is whitin the bounds at position 'i'
        bool withinBounds (double x, int i)
        {
            return (x >= function.lowerBounds[i] && x <= function.upperBounds[i]);
        }

        /// Checks if a entire vector is whitin the bounds
        bool withinBounds (const Vector& v)
        {
            for(int i = 0; i < N; ++i)
                if(!withinBounds(v[i], i))
                    return false;

            return true;
        }



        /// Vector x reached convergence if it is feasible and its fitness is smaller or equal to the optimal value
        inline bool converged (const Vector& x)
        {
            return x.feasible() && (x.fitness <= function.optimal);
        }



        /// These are the bounds handling functions.

        /// If a dimension 'i' is outside the box, set the value of this dimension to the value of the parent
        void conservate (Vector& child, const Vector& parent)
        {
            for(int i = 0; i < N; ++i)
                if(!withinBounds(child[i], i))
                    child[i] = parent[i];
        }

        /// Clip every dimension 'i' to stay in the range:   lower[i] <= x[i] <= upper[i]
        void clip (Vector& child, const Vector&)
        {
            for(int i = 0; i < N; ++i)
                child[i] = std::min(function.upperBounds[i], std::max(function.lowerBounds[i], child[i]));
        }

        /// If any component is outside the box, generate a new random vector
        void reinitialize (Vector& child, const Vector&)
        {
            if(!withinBounds(child))
                child = newVector();
        }



    //private:


        std::vector<int> permutation;   /// Permutation vector

        /// 'std::function' pointing to the bounds handle function
        std::function<void (MDE<FunctionType>*, Vector&, const Vector&)> boundsHandle;



        Population population;   /// Population vector

        Vector best;    /// Best element at any time

        Function function;   /// Function


        ::help::RandInt    randInt;      /// Generate a random integer given an interval
        ::help::RandDouble randDouble;   /// Generate a random real given an interval
    };

} // namespace de


#endif //DE_DPS_DE_DPS_H
