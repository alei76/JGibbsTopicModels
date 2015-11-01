package ml.tm;

import java.util.ArrayList;
import java.util.List;

public class LightLDA extends LDA {

	double[] Qw; //V
	double[][] qw; //qw, V*K
	int[] noSample;

	int[][] Sw; //draw k samples from qw and store them in Sw;
	int MH_step = 2;
	int[][] alias; //   V*K
	double[][] proba; // alias table, generate by Vose's Alias Method,  V*K
	
	
	protected boolean init(LDAarguments option){ // init() -> readFile(filename) -> train()
		if(option == null)
			return false;
		
		alpha = option.alpha;
		beta = option.beta;
		iterations = option.iterations;
		K = option.K;
		V = option.V;
		M = option.M;
		savestep = option.savestep;
		
		vocabulary = new int[V];
		Nmk = new int[M][K];
		Nkt = new int[V][K];
		Nm = new int[M];
		Nk = new int[K];
		z = new int[M][];
		corpus = new int[M][];
		theta = new double[M][K];
		phi = new double[K][V];
		Qw = new double[V];
		qw = new double[V][K];
		noSample = new int[V];
		Sw = new int[V][K];
		alias = new int[V][K];
		proba = new double[V][K];
		
		
		
		return true;
		
	}
	
	public void generateAliasMethod(int w){


		/**
		 * genearte word w's alias table by vose's alias method
		 */

		//compute Qw, qw
		Qw[w] = 0.0;
		for(int k=0;k<K;k++){
//				Qw[w] += alpha * (Nkt[w][k] + beta)/(Nk[k] + beta*V);
				qw[w][k] = (Nkt[w][k] + beta)/(Nk[k] + beta*V);
				Qw[w] += qw[w][k];
			}





		int[] small = new int[K];
		int[] big = new int[K];

		int smallsum = 0;//new int[V];//
		int bigsum = 0;//new int[V];//

		//multiply each probability by K
		double[] q = qw[w];

		for(int j=0;j<K;j++){
				q[j] = q[j] * K / Qw[w];
				if(q[j] < 1){
					small[j] = 1;
					smallsum ++;
				}
				else{

					big[j] = 1;
					bigsum ++;
				}


			}


		//While Small and Large are not empty

			while(smallsum!=0 && bigsum!=0){
				int l = 0;
				int g=0;
				for(int k=0;k<K;k++){
					if(small[k] !=0){
						l = k;
						small[k] = 0;
					}
					if(big[k] !=0){
						g = k;
						big[k] = 0;
					}
				}

				//
				proba[w][l] = q[l];
				alias[w][l] = g;

				q[g] = q[g] + q[l] - 1;
				if(q[g] < 1)
					small[g] = 1;
				else
					big[g] = 1;


				smallsum = 0;bigsum = 0;
				for(int k=0;k<K;k++){
					smallsum += small[k];
					bigsum += big[k];

				}
			}//while



			while(bigsum != 0){

				//Remove the first element from Large; call it g.
				//Set Prob[g]=1.
				int g = 0;
				for(int k=0;k<K;k++){
					if(big[k] !=0){
						g = k;
						big[k] = 0; // remove the element
						break;
					}
				}//for
				proba[w][g] = 1;


				bigsum = 0;
				for(int k=0;k<K;k++){
					bigsum += big[k];

				}
			}//while

			while(smallsum != 0){
				int l = 0;
				for(int k=0;k<K;k++){
					if(small[k] != 0){
						l = k;
						small[k] = 0;
						break;
					}
				}
				proba[w][l] = 1;

				smallsum = 0;
				for(int k=0;k<K;k++){
					smallsum += small[k];
				}
			}//while

	}
	
	public void Estimation(){

		/**
		 *
		 */

		for(int w=0;w<V;w++)
			generateAliasMethod(w);

		for(int iteration=0;iteration<iterations;iteration++){
		//sampling

			if(iteration %100 ==0){
				System.out.println("iteration " + iteration);
				computePhi();
				computeTheta();
				computeLogLikelihood();
			}


			for(int document=0;document<M;document++){

//				int Kd = 0;
				List<Integer> Kdtopic = new ArrayList<Integer>();
				for(int k=0;k<K;k++){
					if(Nmk[document][k] != 0){
//						Kd++;
						Kdtopic.add(k); 
					}
				}


				double betaV = beta * V;
				double sumPd = z[document].length + K*alpha;
			
				for(int w=0;w<z[document].length;w++){

					int real_w = corpus[document][w];
					int currenttopic = z[document][w];
					int oldtopic = currenttopic;
					--Nmk[document][currenttopic];--Nkt[real_w][currenttopic];
					--Nk[currenttopic];   
					
					// MHV to draw new topic
					int new_topic = 0;

					for(int r=0;r<MH_step; r++){
						{
							//Draw a topic from doc-proposal
							double u = Math.random() * sumPd;
							if(u < z[document].length){

								// draw from doc-topic distribution skipping n

								int pos = (int)u;
								new_topic = z[document][pos];

							}else{

								//draw uniformly
								u -= z[document].length;
								u /= alpha;
								new_topic = (int)u;

							}

							if(currenttopic != new_topic){
								//find acceptance probability
								double s = (Nmk[document][currenttopic] + alpha)*(Nkt[real_w][currenttopic] + beta)/(Nk[currenttopic] + betaV);
								double t = (Nmk[document][new_topic] + alpha) * (Nkt[real_w][new_topic] + beta) / (Nk[new_topic] + betaV);

								double oldproba = (currenttopic==oldtopic)?(Nmk[document][currenttopic] +1+alpha):(Nmk[document][currenttopic]+alpha);
								double newproba = (new_topic == oldtopic)?(Nmk[document][new_topic] +1+alpha):(Nmk[document][new_topic]+alpha);
								double accept = (t*oldproba)/(s*newproba);

								// compare against Math.random()
								if(Math.random() < accept)
									currenttopic = new_topic;


							}

						}


					{
						// Draw a topic from word-proposal

						noSample[real_w] ++;
						if(noSample[real_w] > (K/2)){
							generateAliasMethod(real_w);
							noSample[real_w]=0;
						}
						double tmp1 = Math.random();
						double tmp2 = Math.random();
						int die = (int)tmp1*K;
						if(tmp2 < proba[real_w][die])
							new_topic = die;
						else
							new_topic = alias[real_w][die];

						if (currenttopic != new_topic)
						{
							//Find acceptance probability
							double temp_old = (Nmk[document][currenttopic] + alpha) * (Nkt[real_w][currenttopic] + beta) / (Nk[currenttopic] + betaV);
							double temp_new = (Nmk[document][new_topic] + alpha) * (Nkt[real_w][new_topic] + beta) / (Nk[new_topic] + betaV);
							double acceptance =  (temp_new * qw[real_w][currenttopic]) / (temp_old * qw[real_w][new_topic]);

							//Compare against uniform[0,1]
							if (Math.random() < acceptance)
							{
								currenttopic = new_topic;
							}
						}

					}

				}// MH-step


				++Nmk[document][currenttopic];++Nkt[real_w][currenttopic];
				++Nk[currenttopic];
				z[document][w] = currenttopic;

		}// document sampling end


			}

		}
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		LDAarguments arg = new LDAarguments();
		arg.Setalpha(0.1);arg.Setbeta(0.01);
		arg.Setiterations(500);arg.SetK(50);
		arg.SetM(1500);arg.SetV(12419);
		
		
		LightLDA lda = new LightLDA();
		lda.init(arg);
		lda.readFile("nips.txt");
		lda.Estimation();
		
	}

}
