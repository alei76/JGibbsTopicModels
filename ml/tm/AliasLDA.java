/**
 * 
 */
package ml.tm;

import java.util.ArrayList;
import java.util.List;

/**
 * @author hehe
 *
 */
public class AliasLDA extends LDA{

	/**
	 * @param args
	 */
	
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



		for(int w=0;w<V;w++)
			generateAliasMethod(w);

		for(int iteration=0;iteration<iterations;iteration++){
		//sampling


			if(iteration %100 ==0){
				System.out.println("iteration:" + iteration);
				computePhi();
				computeTheta();
//				lda.OutputModel();
				computeLogLikelihood();
			}




			for(int document=0;document<M;document++){
			//compute Pdw and pdw(t)
				List<Integer> Kdtopic = new ArrayList<Integer>();
				for(int k=0;k<K;k++){
					if(Nmk[document][k] != 0){
						Kdtopic.add(k); 
					}
				}

				double betaV = beta * V;

				for(int w=0;w<z[document].length;w++){

					int real_w = corpus[document][w];
					int currenttopic = z[document][w];
					int oldtopic = currenttopic;

					double Pdw = 0;
					double[] pdw = new double[Kdtopic.size()];
					double[] p = new double[Kdtopic.size()];


					--Nmk[document][currenttopic];--Nkt[real_w][currenttopic];
					--Nk[currenttopic];

					for(int k=0;k<Kdtopic.size();k++){
						Pdw += Nmk[document][Kdtopic.get(k)]*(Nkt[real_w][Kdtopic.get(k)] + beta)/(Nk[Kdtopic.get(k)] + betaV);
						p[k] = Pdw;
					}


					double select_pr = Pdw/(Pdw + alpha * Qw[real_w]); //

					//MHV to draw new topic
					int new_topic = 0;
					for(int r=0;r<MH_step;r++){

						//1. Flip a coin
						if(Math.random() < select_pr){
							double u = Math.random() * Pdw;
							for(int k=0;k<Kdtopic.size();k++){
								if(p[k] >= u){
									new_topic = Kdtopic.get(k);
									break;//
								}
							}


						}
						else{
							noSample[real_w] ++;
							if(noSample[real_w] > (K/2)){
								generateAliasMethod(real_w);
								noSample[real_w] = 0;//
							}
							double tmp1 = Math.random();
							double tmp2 = Math.random();
							int die = (int)tmp1*K;
							if(tmp2 < proba[real_w][die])
								new_topic = die;
							else
								new_topic = alias[real_w][die];


						}

						if(currenttopic != new_topic){

							// find the acceptance probability, s->t
							double s = (Nkt[real_w][currenttopic] + beta)/(Nk[currenttopic] + V*beta);
							double t = (Nkt[real_w][new_topic] + beta)/(Nk[new_topic] + V*beta);

							double pi = (Nmk[document][new_topic] + alpha)/(Nmk[document][currenttopic] + alpha)*t/s*(Nmk[document][currenttopic]*s+alpha*qw[real_w][currenttopic])/(Nmk[document][new_topic]*t+alpha*qw[real_w][new_topic]);

							// compare against Math.random()
							if(Math.random() < pi)
								currenttopic = new_topic;
						}


					}//MH_step

					++Nmk[document][currenttopic];++Nkt[real_w][currenttopic];
					++Nk[currenttopic];
					z[document][w] = currenttopic;

				}//document end

			}

		}//iteration end
	}

	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		LDAarguments arg = new LDAarguments();
		arg.Setalpha(0.1);arg.Setbeta(0.01);
		arg.Setiterations(500);arg.SetK(50);
		arg.SetM(1500);arg.SetV(12419);
		
		
		AliasLDA lda = new AliasLDA();
		lda.init(arg);
		lda.readFile("nips.txt");
		lda.Estimation();
		
		
	}

}
