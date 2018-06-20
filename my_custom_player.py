
from sample_players import DataPlayer
from mcts import *
import random

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only *required* method. You can modify
    the interface for get_action by adding named parameters with default
    values, but the function MUST remain compatible with the default
    interface.

    **********************************************************************
    NOTES:
    - You should **ONLY** call methods defined on your agent class during
      search; do **NOT** add or call functions outside the player class.
      The isolation library wraps each method of this class to interrupt
      search when the time limit expires, but the wrapper only affects
      methods defined on this class.

    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    **********************************************************************
    """
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
        import random
        # self.queue.put(random.choice(state.actions()))
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            ###### iterative deepening ######
        # depth_limit = 5
        # for depth in range(1,depth_limit+1):
        #     best_move = self.alpha_beta_search(state,depth)
        # self.queue.put(best_move)
            depth_limit = 5
            for depth in range(1, depth_limit + 1):
                best_move = self.alpha_beta_search(state, depth)
            self.queue.put(best_move)

            #### no iterative deepening ####
            # self.queue.put(self.alpha_beta_search(state))

    def alpha_beta_search(self,state,depth=3):
        """ Return the move along a branch of the game tree that
        has the best possible value.
        """
        def min_value(state, alpha, beta, depth):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), alpha, beta, depth-1))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value

        def max_value(state, alpha, beta, depth):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), alpha, beta, depth-1))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value


        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for action in state.actions():
            value = min_value(state.result(action), alpha, beta, depth-1)
            alpha = max(alpha, value)
            if value >= best_score:
                best_score = value
                best_move = action
        return best_move

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)






class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation
    Monte Carlo Tree Search
    """

    def mcts(self,state):
        root = MCTS_Node(state)
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
