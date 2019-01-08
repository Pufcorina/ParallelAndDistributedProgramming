# MPI programming 
<img src="https://social.microsoft.com/Forums/getfile/754597" align="right"
     title="Size Limit logo by Anton Lovchikov" width="300" height="150">

     MPI is a communication protocol for programming parallel computers. 
     Both point-to-point and collective communication are supported. 
     MPI "is a message-passing application programmer interface, together 
     with protocol and semantic specifications for how its features must 
     behave in any implementation.

## Instalation ##

### MS-MPI Downloads
I have installed the following versions for MS-MPI:

* [MS-MPI v10.0](https://www.microsoft.com/en-us/download/details.aspx?id=57467)
* [Debugger for MS-MPI Applications with HPC Pack 2012 R2](https://www.microsoft.com/en-us/download/details.aspx?id=48215)

### Visual Studio project configuration 

1. File -> New -> Project -> ...
2. Select Visual C++ -> Windows Desktop -> Windows Desktop Wizard
3. Give your project a name and chose a location to store it
4. After that, it will prompt a dialogue window and for our initial "Hello World" project select Empty Project option and click Ok :)
    
    - Just a few steps before we actually write code :) 

5. Go to project properties and now 3 steps, select from configuration properties :
    - C/C++ and in Additional Include Directories select Edit -> New Line -> Browse and select the Include folder from your MPI install location ( for eg. C:\Program Files (x86)\Microsoft SDKs\MPI )
    - now in Additional #using Directories but this time you will select the Lib folder from your MPI install location
    - Linker -> Input and in Additional Dependencies write msmpi.lib and next Ok

6. Now create a source.cpp file and paste the following code:

        #include <iostream>
        #include "mpi.h"

        using namespace std;

        void main(int argc, char **argv) {
            MPI_Init(&argc, &argv);

            cout << "Hello world";

            MPI_Finalize();
        }

7. Run the app and you're set up


## Goals
    The goal of this lab is to implement a distributed algorithm using MPI.

## Requirement
    Perform the multiplication of 2 polynomials, by distributing computation across several nodes using MPI. Use 
    both the regular O(n2) algorithm and the Karatsuba algorithm. Compare the performance with the "regular" CPU 
    implementation from lab 5.

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
| MPI karatsuba   | 11 ms |11 ms |14 ms |13 ms |
| MPI regular   | 21 ms |22 ms |22 ms |20 ms |


    Thought all the tests I’ve put those algorithms to, the results were for the most  part  as  expected, although  
    the  results  may  vary  quite  a  bit  (  see  that up to level 8 the parallelized version of the regular 
    algorithm has the lead, but suddenly  at  level  20  it  takes  quite  a  bit  more time  than  the  sequential  
    one).There are a lot of factors that can be responsible for those inconsistencies, like background processes, 
    memory usage, and the implementation itself.

## Conclusion

* For the most part, the parallelized versions of the algorithms run faster.
* Karatsuba’s is clearly superior to the regular algorithm and for large numbers, it would be preferred
