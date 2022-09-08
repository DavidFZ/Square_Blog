---
title: 2022_7_summary
date: 2022-08-27 11:07:56
tags:
---
> # 最短路
> 
> ***
>
>> ## 图的表示
>>> ### 1.使用二维数组实现邻接矩阵

		开一个大小为 $N \times N$ 的二维数组g。二维数组中的值就表示某两点间的距离。若两点间不连通,则用任意不存在的值来表示不连通，比如无穷大，负数，0。
	
		一般来说点的数量不超过1000时才能使用该方法,当点的数量级到达1e5这个级别后,该方法就ME了。

```cpp
#define N 1005
#define INF 0x3f3f3f3f
int g[N][N];

int n; //点的个数
int m; //边的个数

void init()
{
    //假设用无穷大表示不连通
    memset(g, 0, sizof(g));
    for (int i = 1; i <= n; i++)
        g[i][j] = INF;
}

void build()
{
    //给出m条边,每条边包含的信息有两个顶点和对应的边权
    cin >> n >> m;
    init(); //先初始化
    for (int i = 1; i <= m; i++)
    {
        int u, v, w;
        cin >> u >> v >> w;
        a[u][v] = min(a[u][v], w);
        a[v][u] = min(a[v][u], w); //无向图
    }
}

```

>>> ### 2.2.链式前向星实现邻接表

		依据链表来建图。提高空间的利用率,多用于存储稀疏图。
		·百分之九十九的图都能用该算法实现模型的建立。不排除特殊情况的存在。
```cpp
#define N 100005 //点的最大值
#define M 200005 //边的最大值。如果是无向图边的最大值要开乘以2

int head[N]; //表头，最大值依据点的最大值
int tot;     //边的总数，同时用于给边编号,从1开始。
//假设是无向图,那么正向边为奇数,反向边为偶数,奇数和奇数+1就对应一条边的两个方向。

int n, m; //点的数量和边的数量

struct EDGE
{
    int to;   //该边的另一个点，对应于from
    int next; //指向下一条边
    int val;  //边权
} edge[M];    //依据边的最大值

//在图中加一条边
void add(int from, int to, int val)
{
    edge[++tot].to = to;
    edge[tot].val = val;
    edge[tot].next = head[from];
    head[from] = tot;
}

//图的初始化
void init()
{
    tot = 0;
    memset(head, 0, sizeof(head));
}

//图的建立
void build()
{   
    cin>>n>>m;
    int u, v, w;
    init();
    for (int i = 1; i <= m; i++)
    {
        cin>>u>>v>>w;
        add(u, v, w);
        //add(v, u, w); //适用于无向图
    }
}


//遍历点now的所有邻点
void work(int now)
{
    for (int i = head[now]; i; i = edge[i].next)
    {
        int to = edge[i].to;   //邻点
        int val = edge[i].val; //边权
        //TODO
    }
}


```

>> ## 算法
>>
>>> ### 1.Dijkstra算法

	  适用于不含负权的图，或是稠密图。
	  本质是贪心法+DP，标志是dis[j] = max(dis[j], min(dis[t], con[t][j])); 
	  朴素的n方算法,可以使用堆优化。开三个数组，一个记录连接情况con,一个记录从初始位置到每个点的最短距离,一个记录是否访问过,避免重复访问。

```cpp
#include <iostream>
#include <cstring>
using namespace std;
#define INF 0x3f3f3f3f
#define MAXN 1010
int n, m;            // n个点,m条边
int con[MAXN][MAXN]; //存储每条边
int dis[MAXN];       //存储一号点到每个点的最短距离
bool vis[MAXN];      //存储每个点的最短路是否已经确定


//路径从1开始
void init()
{
    memset(con, 0, sizeof(con));
   //自己到自己无限重量,自己到别的点,默认没路
    memset(vis, 0, sizeof(vis));
}

//求一号点到n号点的最短距离，如果不存在返回-1
int dij(int s)
{
    //memset(dis, 0, sizeof(dis));
    dis[s] = INF;
    for (int i = 1; i < n + 1; i++)
    {
        dis[i] = con[1][i];
    }
    vis[1] = 1;

    for (int i = 0; i < n - 1; i++)
    {
        int t = -1; //在还未确定路的点中，寻找可以载重最大的点
        for (int j = 1; j <= n; j++)
            if (!vis[j] && (t == -1 || dis[t] < dis[j]))
                t = j;
        for (int j = 1; j <= n; j++)
            if (con[t][j] < INF && !vis[j])                   //联通了
                dis[j] = max(dis[j], min(dis[t], con[t][j])); //用t更新其他点的最大重量
        vis[t] = 1;
    }
    return dis[n];
}

int main()
{
    ios::sync_with_stdio(false);
    int num, s, d, t;
    while (cin >> num)
    {
        for (int i = 0; i < num; i++)
        {
            //cout << "Scenario #" << i + 1 << ":" << endl;
            cin >> n >> m;

            init(); //初始化
            for (int i = 0; i < m; i++)
            {
                cin >> s >> d >> t;
                con[s][d] = t;
                con[d][s] = t;
            }

            cout << dij(n) << endl<<endl;
        }
    }

    return 0;
}
```

	堆优化后的部分
	这里使用第一种模板，取负值，等效小顶堆。

```cpp
#define INF 0x3f3f3f3f
typedef pair<int, int> pii;
int n;         //点的数量
int dis[MAXN]; //存储所有点到一号点的距离
int vis[MAXN]; //存储每个点的最短距离是否已经确定
//求一号点到n号点的最短距离，如果不存在，则返回-1
int dij(int s)
{
    memset(dis, INF, sizeof(dis));
    dis[s] = 0;//初始化起始点

    //取负值的堆,等效小顶堆,节约每轮循环找到最短路径的花费
    priority_queue<pii> q;
    q.push({dis[s], s});
    while (!q.empty())
    {
        int now = q.top().second;
        q.pop();

        if (vis[now])//遍历过的不可能再遍历,否则成环
            continue;
        vis[now] = 1;
        
        //遍历与now相邻的所有邻点
        for (int i = head[now]; i; i = edge[i].next)
        {
            int to = edge[i].to;
            int val = edge[i].val;
            if (dis[to] > dis[now] + val)
            {
                dis[to] = dis[now] + val;
                q.push({-dis[to], to});//等效小顶堆
            }
        }
    }
    if (dis[n] == INF)
        return -1;
    return dis[n];
}
```

>>> ### 2.Bellman-Ford & SPFA

		Bellman-Ford 算法是一种用于计算带权有向图中单源最短路径(当然也可以是无向图)。与Dijkstra相比的优点是，也适合存在负权的图。
		若存在最短路(不含负环时)，可用Bellman-Ford求出，若最短路不存在时，Bellman-Ford只能用来判断是否存在负环。
```cpp
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <queue>
#define INF 0x3f3f3f
using namespace std;
const int MAXN = 1e2 + 10;

int n, m, s, tot;
double v;

double dis[MAXN];
int head[MAXN], vis[MAXN];
double cost[MAXN][MAXN], rate[MAXN][MAXN];

struct node
{
    int to;
    int next;
} edge[MAXN * MAXN];

void add(int u, int v)
{
    edge[tot].to = v;
    edge[tot].next = head[u];
    head[u] = tot++;
}

bool SPFA(int s, double v)
{
    for (register int i = 1; i <= n; i++)
    {
        dis[i] = 0;
        vis[i] = 0;
    }

    dis[s] = v;
    vis[s] = 1;
    queue<int> q;
    q.push(s);
    while (!q.empty())
    {
        int x = q.front();
        q.pop();
        vis[x] = 0;
        for (int i = head[x]; i != -1; i = edge[i].next)
        {
            int j = edge[i].to;
            if (dis[j] < (dis[x] - cost[x][j]) * rate[x][j])
            {
                dis[j] = (dis[x] - cost[x][j]) * rate[x][j];
                if (dis[s] > v)//形成正环,ture
                    return true;
                if (!vis[j])//加入队列尝试一下
                {
                    q.push(j);
                    vis[j] = 1;
                }
            }
        }
    }
    return false;
}

int main()
{
    int a, b;
    double cab, rab, cba, rba;
    while (cin >> n >> m >> s >> v)
    {
        memset(cost, 0, sizeof(cost));
        memset(rate, 0, sizeof(rate));
        memset(head, -1, sizeof(head));
        tot = 1;
        for (int i = 0; i < m; i++)
        {
            cin>>a>>b>>rab>>cab>>rba>>cba;
            cost[a][b] = cab;
            cost[b][a] = cba;
            rate[a][b] = rab;
            rate[b][a] = rba;
            add(a, b);
            add(b, a);
        }
        if (SPFA(s, v))
            cout<<"YES"<<endl;
        else
            cout<<"NO"<<endl;
    }
    return 0;
}
```


> # 二分
>
> ***

	  二分算法的基本用途是在单调序列或单调函数中做查找操作，因此问题的答案具有单调性的时候，我们就可以通过二分把求解转换为判定。
	  二分算法的思想是不断将待求解区间平均分成两份，根据求解区间中点的情况来确定目标元素所在的区间，这样就把解的范围缩小一半。

>> ## 快读
>>

		isdigit()在头文件<ctype.h>下,是数字返回非零值(为真)
		还有fread版本的进阶快读,此处略


```cpp
inline int read() //快读
{
    int x = 0, neg = 1;
    char c = getchar();
    while (!isdigit(c))
    {
        if (c == '-')
            neg = -neg //记录正负号
                      c = getchar();
    }
    while (isdigit(c))
    {
        x = 10 * x + c - '0'; //按位读取
        c = getchar();
    }
    return neg * x; //乘上正负号,的到最后的数
}

inline void print(int x) //快写
{
    if (x < 0)
    {
        putchar('-');
        x = -x;
    }
    if (x >= 10)
        print(x / 10);
    putchar(x % 10 + '0');
}
```

>> ## 算法
>> 

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> num;
int count(int dis)
{
    int cnt = 1;
    int next = num[0] + dis;
    for (int i = 1; i < num.size(); i++)
    {
        if (num[i] < next)
            continue;
        cnt++;
        next = num[i] + dis;
    }
    return cnt;
}
int main()
{
    int n, m;
    while (cin >> n >> m)
    {
        num.clear();
        int tmp, nn = n;
        while (nn--)
        {
            cin >> tmp;
            num.push_back(tmp);
        }
        sort(num.begin(), num.end());

        int l = 0;
        int r = num[n - 1] - num[0];
        int t;
        while (l <= r)
        {
            int mid = (l + r) >> 1;
            if (count(mid) >= m)
            {
                t = mid;
                l = mid + 1;
            }
            else
            {
                r = mid - 1;
            }
        }
        cout << t << endl;
    }
    // cout<<num.size()<<endl;
    return 0;
}
```


> # 并查集&最短路
>
> ***
>
> > ## 并查集
> >
> > > ### 操作集
> > >
> > > * 初始化
> > > * 加入节点
> > > * 查询节点
```cpp
#define MAXN 1010;
int par[MAXN];

void init()
{
    for (int i = 1; i <= n; i++)
    {
        par[i] = i;
    }
    //也有一种写法,全部初始化成-1
    return;
}

int find(int x) //递归版
{
    if (par[x] == x) //找到根节点
        return x;
    return par[x] = find(par[x]); //路径压缩
}

int find(int x) //迭代版适用于层数高于三十的,避免栈溢出导致RE
{
    int a = x;
    while (x != parent(x)) //找到根节点
    {
        x = parent(x);
    }

    while (a != parent(a)) //路径压缩
    {
        int z = a;
        a = parent(a);
        parent(z) = x;
    }

    return x;
}

void join(int x, int y) //合并节点
{
    int a = find(x), b = find(y);
    par[a] = b;
}

```
> > > ### 优化
> > > * 按秩合并
> > >
> > > * 状态压缩
> > >
```cpp
#define MAXN 1010;
int par[MAXN];
int rank[MAXN];

void init(int n) // 初始化的过程中加入记录高度的rank数组
{
    for (int i = 0; i < n; i++)
    {
        par[i] = i;
        rank[i] = 0; // 初始树的高度为0
    }
}

void unite(int x, int y) // 按秩合并,避免歪脖子树
{
    x = find(x);
    y = find(y);
    if (x == y)
        return;
    if (rank[x] < rank[y])
        par[x] = y; // 合并是从rank小的向rank大的连边
    else
    {
        par[y] = x;
        if (rank[x] == rank[y]) //一样高则前面的x作为根节点
            ++rank[x];
    }
}

```

> > ## 最短路
> >
>>> ### 0.算法比较

		Prim算法适合稠密图，其时间复杂度为O(n^2)，其时间复杂度与边得数目无关，而Kruskal算法的时间复杂度为O(eloge)跟边的数目有关，适合稀疏图。
		Kruskal算法的数组d[]含义为起点s达到顶点Vi的最短距离,Prim算法的数组d[]含义为顶点Vi与集合S的最短距离
> > > ### 1.Kruskal算法
> > >

		算法原理：
		* 1.将连通网中所有的边按照权值大小做升序排序；
		* 2.从权值最小的边开始选择，只要此边不和已选择的边一起构成环路，就可以选择它组成最小生成树；
		对于 N 个顶点的连通网，挑选出 N-1 条符合条件的边，这些边组成的生成树就是最小生成树。

```cpp
#define MAXN 110
int n, cnt, num, ans, m; // n为点的数量，m是边的总数，num是已经选取了的边数
int par[MAXN];           //存储每个点的父亲

struct Edge //结构体存储数据
{
    int w;        //边权
    int to, from; //边相连的两个点
};
Edge con[1000];

bool cmp(Edge x, Edge y) //使sort按照边权大小排序
{
    return x.w < y.w;
}

int find(int x) //并查集递归版+状态压缩
{
    if (par[x] == x)
        return x;
    return par[x] = find(par[x]);
}
void build(int x) //选取函数，答案加上边权值，然后利用两个点相连
{
    ans += con[x].w;
    par[find(con[x].from)] = find(con[x].to);
    return;
}
void init()
{
    for (int i = 1; i <= n; i++)
        par[i] = i;
    ans = 0;
}
int kruskal()
{
    init(); //辅助参数初始化
    //贪心法灵魂
    sort(con + 1, con + m + 1, cmp); //按边权排序

    for (int i = 1; i < m + 1; i++) //按边权从小到大枚举
    {
        if (num == n - 1)
            break;                                //如果选取的边足够了，就停止循环
        if (find(con[i].from) == find(con[i].to)) //如果已经接入树中，则跳过
            continue;
        else
        {
            build(i);
            num++; //已经选取的边数+1
        }
    }

    return ans;
}
```

> > ### 2.Prim算法
> > 

		实现思路是：
		* 1.将连通网中的所有顶点分为两类（假设为 A 类和 B 类）。初始状态下，所有顶点位于 B 类；
		* 2.选择任意一个顶点，将其从 B 类移动到 A 类；
		* 3. 从 B 类的所有顶点出发，找出一条连接着 A 类中的某个顶点且权值最小的边，将此边连接着的 A 类中的顶点移动到 B 类；
		重复执行第 3  步，直至 B 类中的所有顶点全部移动到 A 类，恰好可以找到 N-1 条边。

```cpp
#define MAXN 1000
#define INF 0x3f3f3f3f

int n, G[MAXN][MAXN];     // n为顶点数，MAXV为最大顶点数
int d[MAXN];              //顶点与集合S的最短距离
bool vis[MAXN] = {false}; //标记数组，vis[i] == true表示访问。初值均为false
int prim()                //默认0号为初始点，函数返回最小生成树的边权之和
{
    // fill(d, d + MAXN, INF);     // fill函数将整个d数组赋为INF
    memset(d, 0, sizeof(d));
    d[0] = 0;                   //只有0号顶点到集合S的距离为0，其余全是INF
    int ans = 0;                //存放最小生成树的边权之和
    for (int i = 0; i < n; i++) //循环n次
    {
        int u = -1, MIN = INF;      // u使d[u]最小，MIN存放该最小的d[u]
        for (int j = 0; j < n; j++) //找到未访问的顶点中d[]最小的
        {
            if (vis[j] == false && d[j] < MIN)
            {
                u = j;
                MIN = d[j];
            }
        }
        //找不到小于INF的d[u]，则剩下的顶点和集合S不连通
        if (u == -1)
            return -1;
        vis[u] = true; //标记u为已访问
        ans += d[u];   //将与集合S距离最小的边加入最小生成树
        for (int v = 0; v < n; v++)
        {
            // v未访问 && u能到达v && 以u为中介点可以使v离集合S更近
            if (vis[v] == false && G[u][v] != INF && G[u][v] < d[v])
            {
                d[v] = G[u][v]; //将G[u][v]赋值给d[v]
            }
        }
    }
    return ans; //返回最小生成树的边权之和
}

```

> # 数论
> 
>>> ## 1.GCD
>>> 

		最大公约数（Greatest Common Divisor，简称GCD）通常使用欧几里得算法，即辗转反除法。
		最小公倍数(Least Common Mutiple, 简称LCM).lcm(a,b) = \frac{a*b}{gcd(a,b)}

```cpp
int gcd(int a, int b) //递归版
{
    return b == 0 ? a : gcd(b, a % b);
}

int lcm(int a, int b)
{
    return (a * b) / gcd(a, b);
}
```

>>> ## 2.素数筛
>>> 


	直接上结论，最优算法，线性筛(欧拉筛)


```cpp
#define MAXN 100100
int m;
bool v[MAXN];
int p[MAXN];
void prime(int n)
{
    m = 0;
    memset(v, 0, sizeof(v));
    for (int i = 2; i < n + 1; i++)
    {
        if (!v[i])
        {
            p[++m] = i;
        }

        for (int j = 1; j <= m && i * p[j] <= n; j++)
        {
            v[i * p[j]] = 1;
            if (i % p[j] == 0)
                break; //这条语句很关键
        }
    }
}
```

>>> ## 3.快速幂 

```cpp
int fast_power(int a, int n, int mod) // a 为底数,n为次数，mod为上限
{
    int ans = 1;
    while (n)
    {
        if (n & 1) //n&1,与运算,可以判断n是否为偶数。如果是偶数，n&1返回0；否则返回1，为奇数。
            ans = ans * a % mod; //二进制拆分n第i位(从右往左)为1
        a = a * a % mod;
        n >>= 1; //等价于 n/=2;
    }
    return ans;
}
```

> # 线段树
>> ## 算法板子
```cpp
#include <bits/stdc++.h>

using namespace std;

const int N = 200010;

int m, p; // 操作数, 取模的数
struct Node
{
    int l, r;
    int v; // 区间[l, r]中的最大值
} tr[N * 4];

void pushup(int u)
{ // 由子节点的信息，来计算父节点的信息
    tr[u].v = max(tr[u << 1].v, tr[u << 1 | 1].v);
}

// 节点tr[u]存储区间[l, r]的信息
void build(int u, int l, int r)
{

    tr[u] = {l, r};
    if (l == r)
        return;
    int mid = l + r >> 1;
    build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
    // pushup(u);  // 不需要这句话，因为线段树中所有的v都是0
}

// 从节点tr[u]开始查询区间[l, r]的最大值
int query(int u, int l, int r)
{

    if (tr[u].l >= l && tr[u].r <= r)
        return tr[u].v; // 树中节点，已经被完全包含在[l, r]中了

    int mid = tr[u].l + tr[u].r >> 1;
    int v = 0;
    if (l <= mid)
        v = query(u << 1, l, r);
    if (r > mid)
        v = max(v, query(u << 1 | 1, l, r)); // 右区间从mid+1开始

    return v;
}

// 从节点tr[u]开始修改第x个数为v
void modify(int u, int x, int v)
{

    if (tr[u].l == x && tr[u].r == x)
        tr[u].v = v;
    else
    {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid)
            modify(u << 1, x, v);
        else
            modify(u << 1 | 1, x, v);
        pushup(u); // 修改子节点，则必须更新父节点
    }
}

int main()
{

    int n, last; // n:插入的数据的个数，last:上次询问的结果
    scanf("%d%d", &m, &p);

    // 创建线段树, 根节点是tr[1]
    build(1, 1, m);

    char op[2]; // 操作码
    int x;      // 操作数
    while (m--)
    {
        scanf("%s%d", op, &x);
        if (*op == 'Q')
        {
            // 从根节点1开始查询区间[n-x+1, n]的最大值
            last = query(1, n - x + 1, n);
            printf("%d\n", last);
        }
        else
        {
            // 从根节点1开始修改第n+1个数为(x + last) % p
            modify(1, n + 1, (x + last) % p);
            n++;
        }
    }

    return 0;
}
```
		hint:父节点 x/2, x>>1;左儿子 2x, x<<1;右儿子， x<<1|1.


> > # 杂项
>
> > ## memset 和 fill 的差别
> >


		因为memset函数按照字节填充，所以一般memset只能用来填充char型数组，(因为只有char型占一个字节)。如果填充int型数组，只能填充0、-1 和 inf(正负都行)。因为00000000 = 0，-1同理，如果我们把每一位都填充“1”，会导致变成填充入“11111111”。如果我们将inf设为0x3f3f3f3f，0x3f3f3f3f的每个字节都是0x3f！所以要把一段内存全部置为无穷大，我们只需要memset(a, 0x3f, sizeof(a))。无穷小可以将-INF设为0x8f。

>> ## fmod()
>> 


		是<math.h>下的C 库函数 double fmod(double x, double y) :返回 x 除以 y 的余数。cpp下为<cmath>。
		可以用来求取浮点数的余数。

```cpp
const double eps = 1e-2;
double gcd(double a, double b)
{
    if (fabs(b) < eps)
        return a;
    if (fabs(a) < eps)
        return b;
    return gcd(b, fmod(a, b));
}
```

>> ## bitset

		<bitset>CPP的STL中的一个库，可以访问指定下标的bit位，还可以把它们作为一个整数来进行某些统计。可以通过数组下标访问！

```cpp
//0. 构造方式，共四种
bitset<4> bitset1;　　//无参构造，长度为４，默认每一位为０
 
bitset<8> bitset2(12);　　//长度为８，二进制保存，前面用０补充
 
string s = "100101";
bitset<10> bitset3(s);　　//长度为10，前面用０补充
  
char s2[] = "10101";
bitset<13> bitset4(s2);　　//长度为13，前面用０补充

//注意:若构造参数比bitsize大，参数为整数时取后面部分，参数为字符串时取前面部分
//bitset<2> bitset1(12);　　//12的二进制为1100，但size为2，只取后面部分，即00


//1. 运算符操作集
bitset<4> foo (string("1001"));
bitset<4> bar (string("0011"));

cout << (foo^=bar) << endl; // 1010 (foo对bar按位异或后赋值给foo)
cout << (foo&=bar) << endl; // 0010 (按位与后赋值给foo)
cout << (foo|=bar) << endl; // 0011 (按位或后赋值给foo)

cout << (foo<<=2) << endl; // 1100 (左移２位，低位补０，有自身赋值)
cout << (foo>>=1) << endl; // 0110 (右移１位，高位补０，有自身赋值)

cout << (~bar) << endl;  // 1100 (按位取反)
cout << (bar<<1) << endl;  // 0110 (左移，不赋值)
cout << (bar>>1) << endl;  // 0001 (右移，不赋值)

cout << (foo==bar) << endl; // false (0110==0011为false)
cout << (foo!=bar) << endl; // true (0110!=0011为true)

cout << (foo&bar) << endl; // 0010 (按位与，不赋值)
cout << (foo|bar) << endl; // 0111 (按位或，不赋值)
cout << (foo^bar) << endl; // 0101 (按位异或，不赋值)


//3. 可用函数
bitset<8> foo ("10011011");

cout << foo.count() << endl;　　//5　　（count函数用来求bitset中1的位数，foo中共有５个１
cout << foo.size() << endl;　　 //8　　（size函数用来求bitset的大小，一共有８位

cout << foo.test(0) << endl;　　//true　　（test函数用来查下标处的元素是０还是１，并返回false或true，此处foo[0]为１，返回true
cout << foo.test(2) << endl;　　//false　　（同理，foo[2]为０，返回false

cout << foo.any() << endl;　　//true　　（any函数检查bitset中是否有１
cout << foo.none() << endl;　　//false　　（none函数检查bitset中是否没有１
cout << foo.all() << endl;　　//false　　（all函数检查bitset中是全部为１

cout << foo.flip(2) << endl;　　//10011111　　（flip函数传参数时，用于将参数位取反，本行代码将foo下标２处"反转"，即０变１，１变０
cout << foo.flip() << endl;　　 //01100000　　（flip函数不指定参数时，将bitset每一位全部取反

cout << foo.set() << endl;　　　　//11111111　　（set函数不指定参数时，将bitset的每一位全部置为１
cout << foo.set(3,0) << endl;　　//11110111　　（set函数指定两位参数时，将第一参数位的元素置为第二参数的值，本行对foo的操作相当于foo[3]=0
cout << foo.set(3) << endl;　　 //11111111　　（set函数只有一个参数时，将参数下标处置为１

cout << foo.reset(4) << endl;　　//11101111　　（reset函数传一个参数时将参数下标处置为０
cout << foo.reset() << endl;　　 //00000000　　（reset函数不传参数时将bitset的每一位全部置为０

//4. 类型转换函数
string s = foo.to_string();　　//将bitset转换成string类型
unsigned long a = foo.to_ulong();　　//将bitset转换成unsigned long类型
unsigned long long b = foo.to_ullong();　　//将bitset转换成unsigned long long类型

cout << s << endl;　　//10011011
cout << a << endl;　　//155
cout << b << endl;　　//155
```

>> ## limits

		C的标准库<limits.h>int的max与min。
		CPP下为头文件为<climits>


