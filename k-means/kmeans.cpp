#include<bits/stdc++.h>
using namespace std;

int n = 1000, d = 2;
int k = 3;

vector< vector<int> > data(n, vector<int>(d));

void createData(){
	for(int i = 0; i < n; i++){
		for(int j = 0; j < d; ++j){
			data[i][j] = rand()%100;
		}
	}
}

vector< vector<int> > selectCenters(){
	vector< vector<int> > ks(k, vector<int>(d));
	for(int i = 0; i < k; i++){
		int row = rand() % n;
		for(int j = 0; j < d; ++j){
			ks[i][j] = data[row][j];
		}
	}
	return ks;
}

vector< vector< vector<int> > > clusterData(vector< vector<int> > ks){
	vector< vector< vector<int> > > cluster(k, vector< vector<int> >(0));
	for(int i = 0; i < n; i++){
		int min = 999, index = 0;
		for(int j = 0; j < k; j++){
			int tmp = 0;
			for(int p = 0; p < d; ++p){
				tmp += (data[i][p] - ks[j][p])*(data[i][p] - ks[j][p]);
			}
			if(tmp < min){
				min = tmp;
				index = j;
			}
		}
		cluster[index].push_back(data[i]);
	}
	return cluster;
}

vector< vector<int> > adjustMean(vector< vector< vector<int> > > cluster, vector< vector<int> > ks){
	int flg = 0;
	while(true){
		for(int i = 0; i < cluster.size(); i++){	//number of clusters
			vector<int> mean(d);
			for(int j = 0; j < cluster[i].size(); ++j){	// points in cluster i
				for(int p = 0; p < cluster[i][j].size(); ++p){
					mean[p] += cluster[i][j][p];
				}
			}
			float diff = 0.0;
			for(int j = 0; j < d; j++){
				diff += abs(ks[i][j] - mean[j] / cluster[i].size());
				ks[i][j] = mean[j] / cluster[i].size();
			}
			if(diff < 1)
				flg = 1;
			else
				flg = 0;
		}
		if(flg == 1)
			break;
	}
	return ks;
}
int predictCluster(vector<int> pt, vector< vector<int> > ks){
	int cno = 0, min = 9999;
	for(int i = 0; i < ks.size(); i++){
		int sum = 0;
		for(int j = 0; j < pt.size(); j++){
			sum += abs(ks[i][j] - pt[i]);
		}
		if(sum < min){
			min = sum;
			cno = i;
		}
	}
	return cno;
}
int main(){
	srand(time(NULL));
	createData();
	vector< vector<int> > ks = selectCenters();
	vector< vector< vector<int> > > clustered = clusterData(ks);
	ks = adjustMean(clustered, ks);
	
	vector<int> pt(d);
	for(int i = 0; i < d; i++)
		pt[i] = rand() % 100;
	cout<<predictCluster(pt, ks);
	
	return 0;
}
