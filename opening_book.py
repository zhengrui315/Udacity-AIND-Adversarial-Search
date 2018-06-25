# develop an opening book of the best moves for every possible game state
# from an empty board to at least a depth of 4 plies

import time
import random, pickle
from collections import defaultdict, Counter

from isolation import Isolation
from isolation.isolation import _WIDTH, _HEIGHT,_ACTIONSET,Action
from _utils import *



def build_table(num_rounds=20):
    # Builds a table that maps from game state -> action
    # by choosing the action that accumulates the most
    # wins for the active player. (Note that this uses
    # raw win counts, which are a poor statistic to
    # estimate the value of an action; better statistics
    # exist.)

    book = defaultdict(Counter)
    for i in range(num_rounds):
        state = Isolation()
        print(i)
        print(state.board,bin(state.board))
        build_tree(state, book)
    return {k: max(v, key=v.get) for k, v in book.items()}


def build_tree(state, book, depth=4):
    if depth <= 0 or state.terminal_test():
        return -simulate(state)
    action = alpha_beta_search(state)
    reward = build_tree(state.result(action), book, depth - 1)

    sym_states_type = symmetric_states(state)
    sym_states = [sym_state[0] for sym_state in sym_states_type]
    for i,sym_state in enumerate(sym_states):
        if sym_state in book.keys():
            sym_type = sym_states_type[i][1]
            sym_action = symmetric_action(action, sym_type)
            book[sym_state][sym_action] += reward
            break
    book[state][action] += reward

    return -reward


def simulate(state):
    while not state.terminal_test():
        state = state.result(random.choice(state.actions()))
    return -1 if state.utility(state.player()) < 0 else 1




###################################################################
###################  symmetries in opening book   #################
###################################################################
def symmetric_positions(position, sym_type):
    row, col = position // (_WIDTH+2), position % (_WIDTH + 2)
    if sym_type == 'LR':
        return row * (_WIDTH+2) + (_WIDTH+1-col)
    elif sym_type == 'UD':
        return (_HEIGHT-1-row) * (_WIDTH+2) + col
    elif sym_type == 'LRUD':
        return (_HEIGHT-1-row) * (_WIDTH+2) + (_WIDTH+1-col)


def symmetric_states(state):
    """  return the states symmetric to the current one   """
    board = bin(state.board)[2:]
    board = [board[i*(_WIDTH+2):(i+1)*(_WIDTH+2)] for i in range(_HEIGHT)]

    # left - right:
    board_1 = eval('0b'+''.join([row[::-1] for row in board]))
    locs_1 = (symmetric_positions(state.locs[0],'LR') if state.locs[0] != None else None,symmetric_positions(state.locs[1],'LR') if state.locs[1] != None else None)
    state_1 = Isolation(board_1, ply_count=state.ply_count, locs=locs_1)
    # up - down:
    board_2 = eval('0b'+''.join(board[::-1]))
    locs_2 = (symmetric_positions(state.locs[0],'UD') if state.locs[0] != None else None,symmetric_positions(state.locs[1],'UD') if state.locs[1] != None else None)
    state_2 = Isolation(board_2, ply_count=state.ply_count, locs=locs_2)

    # left - right and up - down:
    board_3 = eval('0b'+''.join([row[::-1] for row in board[::-1]]))
    locs_3 = (symmetric_positions(state.locs[0],'LRUD') if state.locs[0] != None else None,symmetric_positions(state.locs[1],'LRUD') if state.locs[1] != None else None)
    state_3 = Isolation(board_3, ply_count=state.ply_count, locs=locs_3)

    return [(state_1,'LR'),(state_2,'UD'),(state_3,'LRUD')]


sym_dict_LR = {Action.NNE:Action.NNW,Action.ENE:Action.WNW,Action.ESE:Action.WSW,Action.SSE:Action.SSW,Action.SSW:Action.SSE,Action.WSW:Action.ESE,Action.WNW:Action.ENE,Action.NNW:Action.NNE}
sym_dict_UD = {Action.NNE:Action.SSE,Action.ENE:Action.ESE,Action.ESE:Action.ENE,Action.SSE:Action.NNE,Action.SSW:Action.NNW,Action.WSW:Action.WNW,Action.WNW:Action.WSW,Action.NNW:Action.SSW}
sym_dict_LRUD = {Action.NNE:Action.SSW,Action.ENE:Action.WSW,Action.ESE:Action.WNW,Action.SSE:Action.NNW,Action.SSW:Action.NNE,Action.WSW:Action.ENE,Action.WNW:Action.ESE,Action.NNW:Action.SSE}

def symmetric_action(action, sym_type):
    """ return the symmetric action according to the symmetry type sym_type """

    if action not in _ACTIONSET:
        return symmetric_positions(action,sym_type)
    elif sym_type == 'LR':
        return sym_dict_LR[action]
    elif sym_type == 'UD':
        return sym_dict_UD[action]
    elif sym_type == 'LRUD':
        return sym_dict_LRUD[action]
    else:
        raise ValueError(" The value of sym_type is illegal")





NUM_ROUNDS = 10

if __name__ == "__main__":
    print("Enter>>>")
    start = time.time()
    open_book = build_table(NUM_ROUNDS)
    print(open_book)
    end = time.time()
    print("The total time for {} rounds is {} seconds, the average time for each round is {} seconds per round".format(NUM_ROUNDS,end-start,(end-start)/NUM_ROUNDS))
    with open("data.pickle", 'wb') as f:
        pickle.dump(open_book, f)