using MPI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace PPD_MPI
{
    class MainProgram
    {

        static void listenerThread(Object obj)
        {
            DSM dsm = (DSM) obj;

            while (true)
            {
                Console.WriteLine("Rank " + Communicator.world.Rank.ToString() + " waiting ");
                Msg msg = Communicator.world.Receive<Msg>(Communicator.anySource, Communicator.anyTag);

                if (msg.exit) break;

                if (msg.updateMsg != null)
                {
                    Console.WriteLine("Rank " + Communicator.world.Rank + " received : " + msg.updateMsg.var + " -> " + msg.updateMsg.val);
                    dsm.setVar(msg.updateMsg.var, msg.updateMsg.val);
                }

                if (msg.subscribeMsg != null)
                {
                    Console.WriteLine("Rank " + Communicator.world.Rank + "received: " + msg.subscribeMsg.rank + " sub to " + msg.subscribeMsg.var);
                    dsm.subscribeOther(msg.subscribeMsg.var, msg.subscribeMsg.rank);

                }
                writeVars(dsm);
            }
        }

        static void writeVars(DSM dsm)
        {
            Console.Write("Rank " + Communicator.world.Rank + " a= " + dsm.a + " b= " + dsm.b + " c= " + dsm.c + " subs: ");
            foreach (string var in dsm.subscribers.Keys)
            {
                Console.Write(var + ": [ ");
                foreach (int rank in dsm.subscribers[var])
                {
                    Console.Write(rank + " "); 
                }

                Console.Write("] ");
            }

            Console.WriteLine(); 
        }

        static void Main(string[] args)
        {
            using (new MPI.Environment(ref args))
            {
                DSM dsm = new DSM();

                if (Communicator.world.Rank == 0)
                {

                    Thread thread = new Thread(listenerThread);
                    thread.Start(dsm); 

                    bool exit = false;

                    dsm.subscribeTo("a"); 
                    dsm.subscribeTo("b"); 
                    dsm.subscribeTo("c"); 

                    while (!exit)
                    {
                        Console.WriteLine("1. Set var");
                        Console.WriteLine("2. Change var");
                        Console.WriteLine("0. Exit");

                        int answer;
                        int.TryParse(Console.ReadLine(), out answer);

                        if (answer == 0)
                        {
                            dsm.close(); 
                            exit = true;
                        }else if (answer == 1)
                        {
                            Console.WriteLine("var (a, b, c) = ");
                            string var = Console.ReadLine();

                            Console.WriteLine("val (int) = ");
                            int val;
                            int.TryParse(Console.ReadLine(), out val);

                            dsm.updateVar(var, val);
                            writeVars(dsm);
                        }else if (answer == 2)
                        {
                            Console.WriteLine("var to check (a, b, c) = ");
                            string var = Console.ReadLine();

                            Console.WriteLine("val to check (int) = ");
                            int val;
                            int.TryParse(Console.ReadLine(), out val);

                            Console.WriteLine("val to check (int) = ");
                            int newVal;
                            int.TryParse(Console.ReadLine(), out newVal);

                            dsm.checkAndReplace(var, val, newVal); 
                        }
                    }
                   
                }else if (Communicator.world.Rank == 1)
                {

                    Thread thread = new Thread(listenerThread);
                    thread.Start(dsm);

                    dsm.subscribeTo("a");

                    thread.Join(); 

                }
                else if (Communicator.world.Rank == 2)
                {
                    Thread thread = new Thread(listenerThread);
                    thread.Start(dsm);

                    dsm.subscribeTo("b");

                    thread.Join(); 
                }
            }
        }
    }
}
