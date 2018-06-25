from sample_players import DataPlayer
from mcts import *
from _utils import *
import random

class CustomPlayer_MiniMax(DataPlayer):
    """ Implement customized agent to play knight's Isolation """

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        if state.ply_count < 4:
            if state in self.data:
                self.queue.put(self.data[state])
            else:
                self.queue.put(random.choice(state.actions()))
        else:
            ###### iterative deepening ######
            depth_limit = 5
            for depth in range(1, depth_limit + 1):
                best_move = alpha_beta_search(state, self.player_id, depth)
            self.queue.put(best_move)

            #### no iterative deepening ####
            # self.queue.put(alpha_beta_search(state,self.player_id))



class CustomPlayer_MCTS(DataPlayer):
    """
    Implement an agent to play knight's Isolation with Monte Carlo Tree Search
    """

    def mcts(self, state):
        root = MCTS_Node(state)
        if root.state.terminal_test():
            return random.choice(state.actions())
        for _ in range(iter_limit):
            child = tree_policy(root)
            if not child:
                continue
            reward = default_policy(child.state)
            backup(child, reward)

        idx = root.children.index(best_child(root))
        return root.children_actions[idx]

    def get_action(self, state):
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.mcts(state))


CustomPlayer = CustomPlayer_MiniMax