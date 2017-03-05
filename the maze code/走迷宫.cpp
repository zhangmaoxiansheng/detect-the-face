//�ܽᣬ���ε�ʱ�����鲻��д��ֻд���־�OK�ˣ�Ȼ������������һ��Ҫע�⣬�����ϲ�Ҫ������ 
#include<iostream>
#include<string>
#include<windows.h>
#include<conio.h> 
using namespace std;
 void gotoxy(int x, int y)
{
     COORD cd;
     cd.X = x;
     cd.Y = y;
     HANDLE handle=GetStdHandle(STD_OUTPUT_HANDLE);
     SetConsoleCursorPosition(handle,cd);
     cout<<"Y\b"; 
}
struct coordinate
{
	int x;
	int y;
};
class maze
{
	public:
	    maze(int _maze[8][9])
	    {
	    	int i,j;
	    	for (i = 0;i <= 7;i++)
	    	{
	    	for (j = 0;j <= 8;j++)
	    	m_maze[i][j] = _maze[i][j];
		}
	}
	    void print()
	    {
	    	int i,j;
	    	for (i = 0;i <= 7;i++)
	    	{
	    		for(j = 0;j <= 8;j++)
	    		{if(m_maze[i][j] == 1)cout<<"*";
	    		 if(m_maze[i][j] == 0)cout<<" ";
				}
				cout<<endl;
			}
		}
	   
	private:
		//1����wall��0����ͨ·
		int m_maze[8][9];
	
				 
};
class person
{
	public:
		void setstart(int a,int b)
	    {
	    	start.x = a;
	    	start.y = b;
	    	now.x = start.x;
	    	now.y = start.y;
		}
		void setend(int c,int d)
		{
			end.x = c;
			end.y = d;
		}
		void star()
		{
			gotoxy(start.x,start.y);
		}
		
		
		void move(int _maze[8][9])//���Ĳ����������ⲿ�ģ����ݳ�Ա���ô� 
		{
			char mv;
			mv = getch();//wasd,w��
			cout<<" ";
			if(mv == 'w') now.y = now.y - 1;
			if(mv == 's') now.y = now.y + 1;
			if(mv == 'a') now.x = now.x - 1;
			if(mv == 'd') now.x = now.x + 1;
			if(now.y <= 7 && now.x <= 8 && _maze[now.y][now.x] == 0){gotoxy(now.x,now.y);}
			if(now.y > 7 || now.x > 8 || _maze[now.y][now.x] == 1)
			{
				if(mv == 'w') now.y = now.y + 1;
			    if(mv =='s') now.y = now.y - 1;
			    if(mv == 'a') now.x = now.x + 1;
			    if(mv == 'd') now.x = now.x - 1;
			    gotoxy(now.x,now.y);
			}
		}
        
		void movetoend(int _maze[8][9])
		{
			while(now.x != end.x || now.y != end.y)
			{
				move(_maze);
			}
			cout<<"��"<<endl;
		}
	private:
		coordinate start;
		coordinate end;
		coordinate now;

 }; 
int main()
{
	
	int maze1[8][9] = {
	{1,1,1,1,1,1,1,0,1},
	{1,1,1,1,0,1,1,0,1},
	{1,1,1,1,0,1,1,0,1},
	{1,1,1,1,0,1,1,0,1},
	{1,1,0,0,0,0,1,0,1},
	{1,1,0,1,1,0,0,0,1},
	{1,0,0,1,1,1,1,1,1},
	{1,0,1,1,1,1,1,1,1}
	};
	maze m1(maze1);
	m1.print();
	cout<<"��Ϸ����wsad������������,*����ǽ��Y�ǵ�ǰλ�ã�����shift����ʼ��Ϸ"; 
	person p1;
	p1.setstart(1,7);
	p1.setend(7,0);
	p1.star();
    p1.movetoend(maze1);
	getchar();
	return 0;
}
