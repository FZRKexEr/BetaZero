# include "BoardGame.hpp"

class Gomoku : public BoardGame<Gomoku> {
public:
    static constexpr int BOARD_SIZE = 15;

    Gomoku();

    bool makeMove(int x, int y) override;

    std::vector<std::pair<int, int>> getValidMoves() override;

    void reset() override;
    void pass() override;

    int getGameState() override;
    int getCurrentPlayer() const override;

    uint64_t getHash() const override;
    std::vector<float> toTensor() const override;

    void printBoard() const override;
    std::string getBoardString() const override;

private:
    int board[BOARD_SIZE][BOARD_SIZE]; // -1 未落子, 0 白棋, 1 黑棋

    int currentPlayer;
    uint64_t currentHash;

    // 棋盘状态 -2=未缓存, -1=未结束, 0=白棋胜, 1=黑棋胜, 2=平局
    int gameState;

    static std::array<uint64_t, BOARD_SIZE * BOARD_SIZE> zobristBlack;
    static std::array<uint64_t, BOARD_SIZE * BOARD_SIZE> zobristWhite;
    static uint64_t zobristPlayer;
    static bool zobristInitialized;
    static void initZobrist();

    static inline int getIndex(int x, int y) { return x * BOARD_SIZE + y; }
    static inline std::pair<int, int> getCoordinates(int index) { return {index / BOARD_SIZE, index % BOARD_SIZE}; }

    static const std::array<int, 8> DIRECTION_OFFSETS_X;
    static const std::array<int, 8> DIRECTION_OFFSETS_Y;
};

bool Gomoku::zobristInitialized = false;
std::array<uint64_t, Gomoku::BOARD_SIZE * Gomoku::BOARD_SIZE> Gomoku::zobristBlack;
std::array<uint64_t, Gomoku::BOARD_SIZE * Gomoku::BOARD_SIZE> Gomoku::zobristWhite;
uint64_t Gomoku::zobristPlayer;

// 方向偏移量：左、左上、上、右上、右、右下、下、左下
const std::array<int, 8> Gomoku::DIRECTION_OFFSETS_X = {-1, -1, 0, 1, 1, 1, 0, -1};
const std::array<int, 8> Gomoku::DIRECTION_OFFSETS_Y = {0, -1, -1, -1, 0, 1, 1, 1};

void Gomoku::initZobrist() {
    if (zobristInitialized) {
        return;
    }

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        zobristBlack[i] = dist(gen);
        zobristWhite[i] = dist(gen);
    }

    zobristPlayer = dist(gen);

    zobristInitialized = true;
}

Gomoku::Gomoku() {
    initZobrist();
    reset();
}

void Gomoku::reset() {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            board[i][j] = -1;
        }
    }

    // 黑棋先行
    currentPlayer = 1;

    // 计算初始棋盘哈希值
    currentHash = zobristPlayer;

    // 清空GameState缓存
    gameState = -2;
}

bool Gomoku::makeMove(int x, int y) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE || board[x][y] != -1) {
        return false;
    }

    // 更新棋盘状态
    board[x][y] = currentPlayer;
    currentHash ^= (currentPlayer == 1 ? zobristBlack[getIndex(x, y)] : zobristWhite[getIndex(x, y)]);

    // 切换玩家
    currentPlayer ^= 1;
    currentHash ^= zobristPlayer;

    // 清空GameState缓存
    gameState = -2;

    return true;
}

std::vector<std::pair<int, int>> Gomoku::getValidMoves() {
    std::vector<std::pair<int, int>> validMoves;

    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (board[i][j] == -1) {
                validMoves.push_back({i, j});
            }
        }
    }

    return validMoves;
}

void Gomoku::pass() {
    currentPlayer ^= 1;
    currentHash ^= zobristPlayer;
}

int Gomoku::getGameState() {
    if (gameState != -2) {
        return gameState;
    }

    int validMovesNumber = 0; 
    // 检查是否获胜
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (board[i][j] == -1) {
                validMovesNumber++;
                continue;
            }   
            int player = board[i][j];
            for (int d = 0; d < 4; d++) {
                int count = 1;
                for (int k = 1; k < 5; k++) {
                    int x = i + k * DIRECTION_OFFSETS_X[d];
                    int y = j + k * DIRECTION_OFFSETS_Y[d];
                    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE || board[x][y] != player) {
                        break;
                    }
                    count++;
                }
                if (count >= 5) {
                    return gameState = (player == 1 ? 1 : 0);
                }
            }
        }
    }
    if (validMovesNumber == 0) {
        return gameState = 2;
    }
    return gameState = -1;
}

int Gomoku::getCurrentPlayer() const {
    return currentPlayer;
}

uint64_t Gomoku::getHash() const {
    return currentHash;
}

std::vector<float> Gomoku::toTensor() const {
    int boardArea = BOARD_SIZE * BOARD_SIZE;
    std::vector<float> tensor(3 * boardArea, 0.0f);

    // 第一通道：黑棋位置
    // 第二通道：白棋位置
    // 第三通道：当前玩家平面：若当前玩家为黑（currentPlayer==1），全1；否则全0。
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            int index = getIndex(i, j);
            tensor[index] = board[i][j] == 1 ? 1.0f : 0.0f;
            tensor[boardArea + index] = board[i][j] == 0 ? 1.0f : 0.0f;
            tensor[2 * boardArea + index] = (currentPlayer == 1) ? 1.0f : 0.0f;
        }
    }

    return tensor;
}

void Gomoku::printBoard() const {
    std::cout << getBoardString() << std::endl;
}

std::string Gomoku::getBoardString() const {
    std::ostringstream oss;
    oss << "  ";
    for (int i = 0; i < BOARD_SIZE; i++) {
        oss << char('a' + i) << ' ';
    }
    oss << std::endl;
    
    for (int i = 0; i < BOARD_SIZE; i++) {
        oss << i + 1 << ' ';
        for (int j = 0; j < BOARD_SIZE; j++) {
            oss << (board[i][j] == 1 ? 'X' : (board[i][j] == 0 ? 'O' : '.'));
        }
        oss << std::endl;
    }
    return oss.str();
}   