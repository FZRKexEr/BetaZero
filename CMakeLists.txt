cmake_minimum_required(VERSION 3.27)
project(BetaZero)

set(CMAKE_CXX_STANDARD 17)

add_executable(BetaZero src/main.cpp
        src/chess.cpp
        src/chess.h
        src/Tictac.cpp
        src/Tictac.h
        src/Othello.cpp
        src/Othello.h
        src/MCTS_1.h
        src/MCTS_2.h
        src/MCTS_2_vector.h
)
