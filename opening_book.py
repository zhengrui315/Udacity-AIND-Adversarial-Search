# develop an opening book of the best moves for every possible game state
# from an empty board to at least a depth of 4 plies

from isolation import Isolation
from isolation.isolation import _WIDTH, _HEIGHT

import random, pickle
from collections import defaultdict, Counter


NUM_ROUNDS = 200

def build_table(num_rounds=NUM_ROUNDS):
    # Builds a table that maps from game state -> action
    # by choosing the action that accumulates the most
    # wins for the active player. (Note that this uses
    # raw win counts, which are a poor statistic to
    # estimate the value of an action; better statistics
    # exist.)

    book = defaultdict(Counter)
    for _ in range(num_rounds):
        state = Isolation()
        # print(state.board,bin(state.board))
        build_tree(state, book)
        # print(book)
    return {k: max(v, key=v.get) for k, v in book.items()}

def build_tree(state, book, depth=4):
    if depth <= 0 or state.terminal_test():
        return -simulate(state)
    # action = random.choice(state.actions())
    action = alpha_beta_search(state)
    reward = build_tree(state.result(action), book, depth - 1)
    book[state.board][action] += reward
    return -reward

def simulate(state):
    while not state.terminal_test():
        state = state.result(random.choice(state.actions()))
    return -1 if state.utility(state.player()) < 0 else 1




def alpha_beta_search(state,depth=3):
    """ Return the move along a branch of the game tree that
    has the best possible value.
    """
    def min_value(state, alpha, beta, depth):
        if state.terminal_test():
            return state.utility(state.ply_count % 2)
        if depth <= 0:
            return score(state)
        value = float("inf")
        for action in state.actions():
            value = min(value, max_value(state.result(action), alpha, beta, depth-1))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def max_value(state, alpha, beta, depth):
        if state.terminal_test():
            return state.utility(state.ply_count % 2)
        if depth <= 0: return score(state)
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

def distance(state):
    """ minimum distance to the walls """

    # x_center, y_center = _HEIGHT // 2, (_WIDTH + 2) // 2
    own_loc = state.locs[state.ply_count % 2]
    x_player, y_player = own_loc // (_WIDTH + 2), own_loc % (_WIDTH + 2)
    # return min(x_player, _WIDTH+1-x_player) + min(y_player, _HEIGHT-1-y_player)
    return min(x_player, _WIDTH + 1 - x_player, y_player, _HEIGHT - 1 - y_player)

def score(state):
    own_loc = state.locs[state.ply_count % 2]
    opp_loc = state.locs[1 - state.ply_count % 2]
    own_liberties = state.liberties(own_loc)
    opp_liberties = state.liberties(opp_loc)
    # return len(own_liberties) - len(opp_liberties)
    dis = distance(state)
    if dis >= 2:
        return 2*len(own_liberties) - len(opp_liberties)
    else:
    # states away from walls from be encouraged, so the weight is bigger
        return len(own_liberties) - len(opp_liberties)


if __name__ == "__main__":
    print("Enter>>>")
    open_book = build_table(200)
    print(open_book)
    with open("data.pickle", 'wb') as f:
        pickle.dump(open_book, f)