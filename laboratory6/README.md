# Parallelizing techniques

## Goals
    The goal of this lab is to implement a simple but non-trivial parallel algorithm.

## Requirement
    Perform the multiplication of 2 polynomials. Use both the regular O(n2) algorithm and the Karatsuba algorithm, and each in both the sequencial 
    form and a parallelized form. Compare the 4 variants.

## Computer Specification

    * CPU: Intel Core i7-6700HQ, 2.60GHz
    * RAM: 16 GB
    * System type: 64-bit

## Short Description of the Implementation
    
    Algorithms:
        * Regular polynomial multiplication
        * Karatsuba algorithm

### Regular polynomial multiplication

* Complexity: O(n^2)
* Step  1:  Distribute each term of the to every term of the second polynomial.  Remember that when you multiply two terms together you must multiply the coefficient (numbers) and add the exponents
* Step 2:  Combine like terms (if you can)

### Karatsuba algorithm

* Complexity:  O(n^log3)
* A  fast  multiplication  algorithm  that  uses  a  divide  and conquers approach to multiply two numbers

## Performed tests

    note: by level 'x' i am referring that the algorithms were used to multiply 2 polynomials of rank x and x - 2, 
    with coefficients being random numbers between -10 and 10. And the number of processes is 6.

| Tables                           | Level 5 | Level 8 | Level 15 | Level 20 |
| -------------------------------- |:--------:|:-------:|:---------:|:---------:|
| regular sequential         | 4 ms |  3 ms | 4 ms | 3 ms |
| regular parallelized | 6 ms | 5 ms |5 ms |6 ms |
| karatsuba sequential   | 0 ms |1 ms |0 ms |1 ms |
| karatsuba parallelized   | 0 ms |0 ms |1 ms |0 ms |


    Thought all the tests I’ve put those algorithms to, the results were for the most  part  as  expected, although  
    the  results  may  vary  quite  a  bit  (  see  that up to level 8 the parallelized version of the regular 
    algorithm has the lead, but suddenly  at  level  20  it  takes  quite  a  bit  more time  than  the  sequential  
    one).There are a lot of factors that can be responsible for those inconsistencies, like background processes, 
    memory usage, and the implementation itself.

## Conclusion

* For the most part, the parallelized versions of the algorithms run faster.
* Karatsuba’s is clearly superior to the regular algorithm and for large numbers, it would be preferred
