class GameState:
    def __init__(self, preset):
        self.current_round = 1
        self.max_rounds = preset["total_number_of_rounds"]
        self.required_wins = (self.max_rounds // 2) + 1
        self.players = {
            "you": {
                "wins": 0,
                "sausage": preset["length"]
            },
            "opponent": {
                "wins": 0,
                "sausage": preset["length"]
            }
        }
