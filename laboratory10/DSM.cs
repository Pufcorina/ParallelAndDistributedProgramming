using MPI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PPD_MPI
{
    [Serializable]
    class Msg
    {
        public UpdateMsg updateMsg = null;
        public ChangeMsg changeMsg = null;
        public SubscribeMsg subscribeMsg = null;

        public bool exit = false; 

        public Msg(UpdateMsg updateMsg)
        {
            this.updateMsg = updateMsg;
        }

        public Msg(ChangeMsg changeMsg)
        {
            this.changeMsg = changeMsg;
        }
       
        public Msg(SubscribeMsg subscribeMsg)
        {
            this.subscribeMsg = subscribeMsg; 
        }

        public Msg(bool exit)
        {
            this.exit = exit; 
        }
    }

    [Serializable]
    class SubscribeMsg
    {
        public string var;
        public int rank; 

        public SubscribeMsg(string var, int rank)
        {
            this.var = var;
            this.rank = rank; 
        }
    }

    [Serializable]
    class UpdateMsg
    {
        public string var;
        public int val;

        public UpdateMsg(string var, int val)
        {
            this.var = var;
            this.val = val; 
        }
    }

    [Serializable]
    class ChangeMsg
    {
        public string var;
        public int oldVal;
        public int newVal; 
        
        public ChangeMsg(string var, int oldVal, int newVal)
        {
            this.var = var;
            this.oldVal = oldVal;
            this.newVal = newVal; 
        }
    }

    class DSM
    {
        public int a = 1, b = 2, c = 3;
        public Dictionary<String, List<int>> subscribers = new Dictionary<string, List<int>>(); 

        public DSM() {
            subscribers.Add("a", new List<int>());
            subscribers.Add("b", new List<int>());
            subscribers.Add("c", new List<int>());
        }

        public void updateVar(string var, int val)
        {

            this.setVar(var, val); 
            UpdateMsg updateMsg = new UpdateMsg(var, val);
            Msg msg = new Msg(updateMsg);

            this.sendToSubscribers(var, msg); 
        }

        public void close()
        {
            this.sendAll(new Msg(true)); 
        }

        public void sendAll(Msg msg)
        {
            for (int i = 0; i < Communicator.world.Size; i++)
            {
                if (Communicator.world.Rank == i) continue;
                Communicator.world.Send(msg, i, 0);
            }
        }

        public void setVar(string var, int val)
        {
            if (var == "a") a = val; 
            if (var == "b") b = val; 
            if (var == "c") c = val; 
        }

        public void subscribeTo(string var)
        {
            this.subscribers[var].Add(Communicator.world.Rank);

            this.sendAll(new Msg(new SubscribeMsg(var, Communicator.world.Rank))); 
        }

        public void subscribeOther(string var, int rank)
        {
            this.subscribers[var].Add(rank); 
        }

        public void sendToSubscribers(string var, Msg msg)
        {
            for (int i = 0; i < Communicator.world.Size; i++)
            {
                if (Communicator.world.Rank == i) continue;
                if (!isSubscribedTo(var, i)) continue; 

                Communicator.world.Send(msg, i, 0);
            }
        }

        public bool isSubscribedTo(string var, int rank)
        {
            if (subscribers[var].Contains(rank))
            {
                return true;
            }

            return false; 
        }

        internal void checkAndReplace(string var, int val, int newVal)
        {
           if (var == "a")
            {
                if (a == val)
                {
                    updateVar("a", newVal);
                }
            }

            if (var == "b")
            {
                if (b == val)
                {
                    updateVar("b", newVal);
                }
            }

            if (var == "c")
            {
                if (a == val)
                {
                    updateVar("c", newVal);
                }
            }
        }
    }
}
