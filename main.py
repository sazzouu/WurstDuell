from scipy.optimize import linprog
from Game import GameState
from copy import deepcopy as copy
import numpy as np

use_preset = True
preset = {
    "total_number_of_rounds": 5,
    "length": 100
}

game_results = list()


def find_best_strategy(game):
    print("Current Round: {0}\nSpielstand: ({1}, {2})".format(game.current_round, game.players["you"]["wins"], game.players["opponent"]["wins"]))
    m = game.players["opponent"]["sausage"]
    x = game.players["you"]["sausage"]
    m_required_wins = game.required_wins - game.players["opponent"]["wins"]
    x_required_wins = game.required_wins - game.players["you"]["wins"]

    if game_results[game.players["you"]["wins"]][game.players["opponent"]["wins"]][x - 1][m - 1]:
        return game_results[game.players["you"]["wins"]][game.players["opponent"]["wins"]][x - 1][m - 1]

    if m_required_wins == 0:
        print("Gegner hat gewonnen!")
        my_strategy = np.zeros(x)
        my_strategy[x - 1] = 1
        opponent_strategy = np.zeros(m)
        opponent_strategy[m - 1] = 1
        game_results[game.players["you"]["wins"]][game.required_wins][x - 1][m - 1] = (my_strategy, opponent_strategy, (-1, 1))
        game_results[game.required_wins][game.players["you"]["wins"]][m - 1][x - 1] = (opponent_strategy, my_strategy, (1, -1))
        return my_strategy, opponent_strategy, (-1, 1)

    if x_required_wins == 0:
        print("Du hast gewonnen!")
        my_strategy = np.zeros(x)
        my_strategy[x - 1] = 1
        opponent_strategy = np.zeros(m)
        opponent_strategy[m - 1] = 1
        game_results[game.required_wins][game.players["opponent"]["wins"]][x - 1][m - 1] = (my_strategy, opponent_strategy, (1, -1))
        game_results[game.players["opponent"]["wins"]][game.required_wins][m - 1][x - 1] = (opponent_strategy, my_strategy, (-1, 1))
        return my_strategy, opponent_strategy, (1, -1)

    if game.current_round == game.max_rounds:
        print("We've reached the final round...")
        payoff = 1
        if game.players["you"]["sausage"] < game.players["opponent"]["sausage"]:
            print("You've lost the game...")
            payoff = -1
        else:
            print("You've won the game!")


        my_strategy = np.zeros(x)
        my_strategy[x - 1] = 1
        opponent_strategy = np.zeros(m)
        opponent_strategy[m - 1] = 1
        game_results[game.required_wins][game.required_wins][x - 1][m - 1] = (my_strategy, opponent_strategy, (payoff, -1 * payoff))
        game_results[game.required_wins][game.required_wins][m - 1][x - 1] = (opponent_strategy, my_strategy, (-1 * payoff, payoff))
        return my_strategy, opponent_strategy, (payoff, -1 * payoff)

    must_keep = game.max_rounds - game.current_round
    # Building Payoff-Matrix
    payoffs = np.ones(shape=(m - must_keep, x - must_keep))

    opponent_strategy = np.zeros(m - must_keep)
    my_strategy = np.zeros(x - must_keep)
    if x * m_required_wins < m:
        print("Opponent wins instantly")
        opponent_strategy[x - 1] = 1.0
        my_strategy[-1] = 1.0
        game_results[game.players["you"]["wins"]][game.players["opponent"]["wins"]][x - 1][m - 1] = (my_strategy, opponent_strategy, (-1, 1))
        game_results[game.players["opponent"]["wins"]][game.players["you"]["wins"]][m - 1][x - 1] = (opponent_strategy, my_strategy, (1, -1))
        return my_strategy, opponent_strategy, (-1, 1)
    elif m * x_required_wins < x:
        print("You win instantly")
        my_strategy[m - 1] = 1.0
        opponent_strategy[-1] = 1.0
        game_results[game.players["you"]["wins"]][game.players["opponent"]["wins"]][x - 1][m - 1] = (my_strategy, opponent_strategy, (1, -1))
        game_results[game.players["opponent"]["wins"]][game.players["you"]["wins"]][m - 1][x - 1] = (opponent_strategy, my_strategy, (-1, 1))
        return my_strategy, opponent_strategy, (1, -1)
    else:
        for row in range(0, m - must_keep):
            for column in range(0, x - must_keep):
                new_game = copy(game)
                new_game.current_round += 1
                new_game.players["opponent"]["sausage"] -= row + 1
                new_game.players["you"]["sausage"] -= column + 1

                if row > column:
                    new_game.players["opponent"]["wins"] += 1
                else:
                    new_game.players["you"]["wins"] += 1

                (_, _, payoff) = find_best_strategy(new_game)
                payoffs[row, column] = payoff[0]

        linear_offset = -2
        result = linprog(c=np.ones(m - must_keep),
                         A_ub=(payoffs.transpose() * -1) + linear_offset, b_ub=np.ones(x - must_keep) * -1,
                         A_eq=[np.ones(m - must_keep)], b_eq=[1],
                         bounds=[(0, 1) for _ in range(0, m - must_keep)])
        opponent_strategy = result["x"]

        result = linprog(c=np.ones(x - must_keep),
                         A_ub=payoffs + linear_offset, b_ub=np.ones(m - must_keep),
                         A_eq=[np.ones(x - must_keep)], b_eq=[1],
                         bounds=[(0, 1) for _ in range(0, x - must_keep)])
        my_strategy = result["x"]

        expected_payoff = 0
        for i in range(0, x - must_keep):
            for j in range(0, m - must_keep):
                expected_payoff += my_strategy[i] * opponent_strategy[j] * payoffs[j][i]
        game_results[game.players["you"]["wins"]][game.players["opponent"]["wins"]][x - 1][m - 1] = (my_strategy, opponent_strategy, (expected_payoff, -1 * expected_payoff))
        game_results[game.players["opponent"]["wins"]][game.players["you"]["wins"]][m - 1][x - 1] = (opponent_strategy, my_strategy, (-1 * expected_payoff, expected_payoff))
        return my_strategy, opponent_strategy, (expected_payoff, -1 * expected_payoff)


if __name__ == '__main__':
    game_results = [[[[None for m in range(0, preset["length"])] for n in range(0, preset["length"])] for l in range(0, preset["total_number_of_rounds"] // 2 + 2)] for w in range(0, preset["total_number_of_rounds"] // 2 + 2)]

    game = GameState(preset)
    (
        game.players["you"]["strategy"],
        game.players["opponent"]["strategy"],
        game.expected_payoffs
    ) = find_best_strategy(game)

    export = open("./export.tsv", "w")
    for w in range(0, preset["total_number_of_rounds"] // 2 + 1):
        for l in range(0, preset["total_number_of_rounds"] // 2 + 1):
            for n in range(0, preset["length"]):
                for m in range(0, preset["length"]):
                    if game_results[w][l][n][m]:
                        result = game_results[w][l][n][m]
                        print("Exporting result for round {0} with ({1}, {2})".format(w + l + 1, w, l))
                        strategy = result[0]
                        for i in range(0, preset["length"]):
                            if i < strategy.size:
                                export.write(str(strategy[i]).replace(".", ",") + "\t")
                            else:
                                export.write("\t")

                        strategy = result[1]
                        for i in range(0, preset["length"]):
                            if i < strategy.size:
                                export.write(str(strategy[i]).replace(".", ",") + "\t")
                            else:
                                export.write("\t")

                        export.write(str(result[2][0]).replace(".", ",") + "\t")
                        export.write(str(result[2][1]).replace(".", ",") + "\t")
                        export.write(str(w) + "\t" + str(l) + "\t" + str(n) + "\t" + str(m))
                        export.write("\n")
    export.close()
