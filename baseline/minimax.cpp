//
//  main.cpp
//  茉莉蜜茶
//
//  Created by LiZnB on 2021/1/15.
//

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <stdlib.h>
#include <string>
#include <assert.h>
#include <vector>
#define N (1 << 20)
#define INF (0x3f3f3f3f)
#define pos(x, y) (1llu << ((x) * 8 + (y)))
#define DEP 6
int DEEP;
#define TIME (0.9)
//#define _BOTZONE_ONLINE


using namespace std;


int dx[] = {0, 1, 1, 1, 0, -1, -1, -1};
int dy[] = {1, 1, 0, -1, -1, -1, 0, 1};

struct Board {
    unsigned long long col[2];

    inline bool operator == (const Board &oth) {
        if (col[1] != oth.col[1]) return false;
        if (col[0] != oth.col[0]) return false;
        return true;
    }
    
 inline int get_color(int x, int y) { // O(1) 
  if (col[1] & pos(x, y)) return 1;
  if (col[0] & pos(x, y)) return 0;
  return -1;
 }

 inline int interval(int x, int y, int dir, int o) { // O(n)
  if (x >= 8 || y >= 8) return -INF;
  if (x < 0 || y < 0) return -INF;

  int color = get_color(x, y);
  
  if (color == -1) return -INF;
  if (color == o) return 0;
  return interval(x + dx[dir], y + dy[dir], dir, o) + 1;
 }

 inline bool can_down(int x, int y, int o) { // O(n ^ 2)
  if (col[1] & pos(x, y)) return false;
  if (col[0] & pos(x, y)) return false;
  for (int i = 0; i < 8; i++) {
   if (interval(x + dx[i], y + dy[i], i, o) >= 1)
    return true;
  }
  return false;
 }

 inline int end() { // O(n ^ 4)
  int num_B = 0, num_W = 0;
 
  for (int i = 0; i < 8; i++) {
   for (int j = 0; j < 8; j++) {
    if (can_down(i, j, 1)) return -1; // 未完
    if (can_down(i, j, 0)) return -1;
    if (get_color(i, j) == 1) num_B++;
    if (get_color(i, j) == 0) num_W++;
   }
  }
  if (num_W == num_B) return 2; // 平局
  return num_W > num_B ? 0 : 1; 
 }

 void init() { // O(1)
        col[1] = col[0] = 0llu;
  col[0] |= (pos(3, 3) | pos(4, 4));
  col[1] |= (pos(3, 4) | pos(4, 3));
 }
    
    inline void down(int x, int y, int o) { // O(n ^ 2)
        col[o] |= pos(x, y);
        for (int i = 0; i < 8; i++) {
            int len = interval(x + dx[i], y + dy[i], i, o);
            int nx = x, ny = y;
            for (int j = 1; j <= len; j++) {
                nx += dx[i];
                ny += dy[i];
                col[o ^ 1] -= pos(nx, ny);
                col[o] |= pos(nx, ny);
            }
        }
    }
    
    bool can_down_all(int o) { // O(n ^ 4)
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (can_down(i, j, o)) return true;
            }
        }
        return false;
    }
};


void draw(Board m, int o) {
    system("clear");
    printf(" ");
    for (int i = 0; i < 8; i++)
        printf(" %c", 'a' + i);
    cout << endl;
    for (int i = 0; i < 8; i++) {
        printf("%c", '1' + i);
        for (int j = 0; j < 8; j++) {
            int color = m.get_color(i, j);
            if (color == -1) {
                cout << " +";
            } else if (color) {
                cout << " ○";
            } else {
                cout << " ●";
            }
        }
        printf("\n");
    }
    cout << "当前该" << (o ? "白棋" : "黑棋") << "落子" << endl;
}

struct HASH {
 Board key;
 int value, color;
    bool exist;
    HASH() {
        exist = false;
    }
} Hash[N];

const unsigned long long Zobrist[8][8][2] {
    10467073703097303640llu, 14028690400316088321llu, 12185173808799298368llu, 9505892039346373052llu,
    16079603106511194126llu, 14343635169364920416llu, 17428882061482346960llu, 16820248244311839879llu,
    12445509553114115028llu, 17502362799816767701llu, 14595085222060672238llu, 14942073406586005928llu,
    16466404002092825192llu, 15576618825824377068llu, 16023949123928518495llu, 9315458112176850579llu,
    14842780254051212665llu, 11142165538995601365llu, 16883505631172262687llu, 17456261295215587166llu,
    10107341262646655873llu, 16569438194165648202llu, 14104250906140892506llu, 9640050574723241242llu,
    16888655521214468910llu, 12626791601073546085llu, 13600748902597824287llu, 12540039884415634825llu,

15727334502064093580llu, 9696321413558542057llu, 14369250330285917935llu, 15295778085173907899llu,
    17819054724123428098llu, 12053950387378321883llu, 17414269728251597809llu, 13499578406393967722llu,
    12426003888742080535llu, 12718089724099860889llu, 18263963131057587203llu, 17582366387579182904llu,
    18087398915800844617llu, 10205522570517524876llu, 12051974110204889561llu, 9905828719297870016llu,
    16469753336030692034llu, 18250332196346333888llu, 10115696361437008107llu, 13020232366663021371llu,
    18246298552968536133llu, 16062523835453117602llu, 17837934720926389608llu, 9551845011122622305llu,
    16119078287380810429llu, 15584654318088079272llu, 17099571918572559668llu, 13918237618398284567llu,
    12705111399145838091llu, 12352707761387328850llu, 17813436719146787623llu, 12762755234933865925llu,
    15897600821220655202llu, 10513265113382260769llu, 16749711707361830375llu, 15881453467100393716llu,
    16695084053103717245llu, 13985150397274565919llu, 13479775656995085879llu, 18436079790902193848llu,
    11247585418432646949llu, 11031928830283320432llu, 15180139342471286852llu, 16458827818388527243llu,
    13634857874459940829llu, 11955625029846587491llu, 15312231614814809141llu, 14321185979515218200llu,
    11054995307419223707llu, 17774411849763262045llu, 14807523146034501107llu, 15278806758334930847llu,
    11310280102805818809llu, 9230808875108664506llu, 17357058122706395146llu, 14242699494417424440llu,
    10718068307775060308llu, 11805926714021293023llu, 14157221877495326155llu, 12613502652921831094llu,
    11962310040726405700llu, 15744548804057420063llu, 10551965962080499392llu, 9917555597505411132llu,
    17704086781260599375llu, 12069809836732461909llu, 17318141241239453567llu, 13643276055377688119llu,
    18117983253392713379llu, 13672379349183530048llu, 14418642478413949522llu, 17481254473138243360llu,
    14772711817207189702llu, 16827437431473298700llu, 13299569474402557488llu, 9339569082119894617llu,
    13469079693882012503llu, 10945375332670052268llu, 9693242241638860877llu, 12741763938421793537llu,
    16825114604893847524llu, 14214910766169989256llu, 11298251728574713609llu, 11622502050736912754llu,
    10564233063191378540llu, 11549863119808455202llu, 11503622001258603632llu, 10683585651548947127llu,
    11441080416409975065llu, 16080224343805310058llu, 12028952642558564739llu, 11732023004494293104llu,
    13332616186683233987llu, 12827349626106907800llu, 11671979762261199758llu, 12485196182060357222llu,
    13689111941923901085llu, 17837371979388935678llu, 16258859383458614874llu, 17136715158445507681llu
};


int tx, ty;

int weight[8][8] = {
    20, -3, 11,  8,  8, 11, -3, 20,
    -3, -7, -4,  1,  1, -4, -7, -3,
    11, -4,  2,  2,  2,  2, -4, 11,
     8,  1,  2, -3, -3,  2,  1,  8,
     8,  1,  2, -3, -3,  2,  1,  8,
    11, -4,  2,  2,  2,  2, -4, 11,
    -3, -7, -4,  1,  1, -4, -7, -3,
    20, -3, 11,  8,  8, 11, -3, 20,
};

inline int H(Board *m) { // O(n ^ 4)
    // 行动力
    // 散度
    // 凝聚手
    // 估值表
    int res = 0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int t = m->get_color(i, j);
            if (t == -1) {
                if (m->can_down(i, j, 1)) res += 5;
                if (m->can_down(i, j, 0)) res -= 5;
                continue;
            }
            res += (t ? 1 : -1) * weight[i][j];
        }
    }
    return res;
}

double st, ed;
int cnt = 0;
vector <int> que;

int DFS(Board m, int o, int dep, int AB, unsigned long long compress) {
    int result = m.end();
    int trans = compress & (N - 1);
    ed = clock();
    if ((ed - st) / CLOCKS_PER_SEC > TIME) {
        return H(&m);
    }
    
    if (result != -1) {
        if (result == 2) return 0;
        return result ? INF / 2 : -INF / 2;
    }
    
    if (Hash[trans].exist == true && Hash[trans].key == m && Hash[trans].color == o) {
        cnt++;
        return Hash[trans].value;
    }
    que.push_back(trans);
    Hash[trans].exist = true;
    Hash[trans].key = m;

    
    if (dep == DEEP + 1) {
        int res = H(&m);
        Hash[trans].color = o;

return Hash[trans].value = res;
    }
    // 启发式搜索 开64位的表，做一次一层的估值？？
    Board sta;
    int best = -INF, res, x = -1, y = -1;
    
    for (int i = 0; i < 8; i++) { // O(n ^ 6)
        for (int j = 0; j < 8; j++) {
            if (m.can_down(i, j, o) == false) continue;
            sta = m;
            sta.down(i, j, o);
            if (sta.can_down_all(o ^ 1)) // 对面可以落子
                res = DFS(sta, o ^ 1, dep + 1, best, compress ^ Zobrist[i][j][o]);
            else // 对面不能落子
                res = DFS(sta, o, dep + 1, -INF, compress ^ Zobrist[i][j][o]);
            if (o && res > best) {
                best = res;
                x = i; y = j;
            } else if (!o && -res > best) {
                best = -res;
                x = i; y = j;
            }
            if (-best <= AB) {
                Hash[trans].exist = false;
                return o ? best : -best;
            }
            ed = clock();
            if ((ed - st) / CLOCKS_PER_SEC > TIME) {
                Hash[trans].exist = false;
                tx = x;
                ty = y;
                return o ? best : -best;
            }
        }
    }
    
    tx = x;
    ty = y;
    Hash[trans].color = o;
    return Hash[trans].value = o ? best : -best;
}

void run_PC(Board *m, int o) {
    
    
    unsigned long long compress = Zobrist[3][3][0] ^ Zobrist[4][4][0];
    compress ^= Zobrist[3][4][1] ^ Zobrist[4][3][1];
    DEEP = 6;
    unsigned long size = que.size();
    for (int i = 0; i < size; i++) {
        Hash[que[i]].exist = false;
    }
    que.clear();
    DFS(*m, o, 1, -INF, compress);
  //  printf("(tx: %d ty:%d)\n", tx, ty);
    
    m->down(tx, ty, o);
}

void run_Player(Board *m, int o) {
    string s;
    int x, y;
    do {
        cin >> s;
        y = s[0] - 'a';
        x = s[1] - '1';
        printf("你选择了%d %d\n", x, y);
    } while (m->can_down(x, y, o) == false);
    m->down(x, y, o);
}

int main(void) {
    st = clock();
    
    Board m;
    int o = 1;
    int PC = 1;
    
    m.init();
    
#ifdef _BOTZONE_ONLINE
    int n, x, y;
    
    cin >> n;
    for (int i = 1; i <= 2 * n - 1; i++) {
        cin >> x >> y;
        if (i == 1) {
            if (x == -1 && y == -1) {
                PC = 0;
            } else {
                PC = 1;
            }
        }
        if (x == -1 && y == -1) {
            PC ^= 1;
            continue;
        }
        m.down(x, y, PC);
        PC ^= 1;
    }

    if (m.can_down_all(PC) == 0) {
        printf("-1 -1\n");
        printf("失败");
        printf("data\n");
        printf("globaldata\n");
    } else {
        run_PC(&m, PC);
        printf("%d %d\n", tx, ty);
        puts("no");
        printf("data\n");
        printf("globaldata\n");
    }
    char s[100];
    scanf("%s %s", s, s);
#endif
    
#ifndef _BOTZONE_ONLINE
cout << "请选择执黑(0)还是执白(1): ";
cin >> PC;
o = 1; // 游戏开始白棋先行

while (m.end() == -1) {
    draw(m, o ^ 1);
    if (m.can_down_all(o) == false) {
        cout << (o ? "黑棋" : "白棋") << "没有落子的地方" << endl;
        o ^= 1;
        continue;
    }
    cnt = 0;
    st = clock();
    (o == PC) ? run_PC(&m, o) : run_Player(&m, o);
    ed = clock();
    if (o == PC) {
        printf("AI思考耗时: %.3lfs 缓存命中: %d\n", (ed - st) / CLOCKS_PER_SEC, cnt);
    }
    o ^= 1;
}
draw(m, o ^ 1);

// 显示游戏结果
int result = m.end();
if (result == 2) {
    cout << "游戏结束，双方平局！" << endl;
} else if (result == PC) {
    cout << "游戏结束，AI获胜！" << endl;
} else {
    cout << "游戏结束，恭喜你获胜！" << endl;
}                                    

#endif
    
    return 0;
}