package ml.tm;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class LDA {
	
	int iterations; //Gibbs sampling iterations
	int savestep;
	double alpha; //hyperparameter
	double beta; //hyperparameter
	int K; // the number of topics
	int V; // the number of words
	int M;//the number of documents
	int[] vocabulary; // the vocabulary
	int[][] Nmk; // each document's topic number
	int[][] Nkt; // each word assigned to each topic number
	int[] Nm; //the number of words in document d
	int[] Nk; // the number of words assigned to topic k
	int[][] z; 
	int[][] corpus; //
	double[][] theta;
	double[][] phi;
	long startTime=System.currentTimeMillis();
	
	public LDA(){
		setDefaultValues(); // call set default method
	}
	
	public void setDefaultValues(){
		
		M = 0;
		V = 0;
		K = 0;
		alpha = 50.0 / K;
		beta = 0.01;
		iterations = 500;
		savestep = 50;
		
		vocabulary = null;
		Nmk = null;
		Nkt = null;
		Nm = null;
		Nk = null;
		z = null;
		corpus = null;
		theta = null;
		phi = null;
	}

	public void readFile(String filename){
		
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String line = null;
			int count = 0;
			try {
				br.readLine();br.readLine();
				while((line = br.readLine()) != null){

					String[] data = line.split(" ");
					z[count] = new int[data.length];
					corpus[count] = new int[data.length];
					Random rand = new Random();
					for(int i=0;i<data.length;i++){

						int tmpz = rand.nextInt(K);//random initialize topic
						corpus[count][i] = Integer.valueOf(data[i])-1;//store input data into corpus
						z[count][i] = tmpz;
						Nmk[count][tmpz] ++;
						Nkt[Integer.valueOf(data[i])-1][tmpz] ++;
						Nm[count] ++;
						Nk[tmpz] ++;

					}
					count ++;
				}



			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}


	}
	
	public void saveModel(){
		
		String prefix = String.valueOf(iterations) + "iterations" + "_K" + K + "_alpha" + alpha+"_beta" + beta;
		try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(prefix + "Nmk.txt"));
			for(int m=0;m<M;m++){
				for(int k=0;k<K;k++){
					bw.write(Nmk[m][k] + " ");
				}
				bw.write("\n");
			}
			bw.flush();bw.close();

			bw = new BufferedWriter(new FileWriter(prefix + "Nkt.txt"));

			for(int v=0;v<V;v++){
				for(int k=0;k<K;k++){
					bw.write(Nkt[v][k] + " ");
				}
				bw.write("\n");
			}
			bw.flush();bw.close();

			bw = new BufferedWriter(new FileWriter(prefix + "Nm.txt"));
			for(int m=0;m<M;m++)
				bw.write(Nm[m] + "\n");
			bw.flush();bw.close();


			bw = new BufferedWriter(new FileWriter(prefix + "Nk.txt"));
			for(int k=0;k<K;k++)
				bw.write(Nk[k] + "\n");
			bw.flush();bw.close();

			bw = new BufferedWriter(new FileWriter(prefix + "z.txt"));
			for(int m=0;m<M;m++){
				for(int w=0;w<z[m].length;w++)
					bw.write(z[m][w] + " ");
				bw.write("\n");
			}
			bw.flush();bw.close();

			bw = new BufferedWriter(new FileWriter(prefix + "phi.txt"));
			for(int k=0;k<K;k++){
				for(int v=0;v<V;v++)
					bw.write(phi[k][v] + " ");
				bw.write("\n");
			}
			bw.flush();bw.close();

			bw = new BufferedWriter(new FileWriter(prefix + "theta.txt"));
			for(int m=0;m<M;m++){
				for(int k=0;k<K;k++)
					bw.write(theta[m][k] + " ");
				bw.write("\n");
			}
			bw.flush();bw.close();

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	
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
		
		return true;
		
	}
	
	
	public void Estimation(){
		
		/**
		 *
		 */

		for(int iteration=0;iteration<iterations;iteration++){

			//sampling

			
			if(iteration %100 ==0){
				System.out.println("iteration: " + iteration);
				computePhi();
				computeTheta();
//				lda.OutputModel();
				computeLogLikelihood();
			}
			for(int document=0;document<M;document++){
				for(int i=0;i<z[document].length;i++){
					int word = corpus[document][i];
					int tmpz = gibbsampling(document, i);//gibbs sampling
					Nmk[document][tmpz] ++;
					Nkt[word][tmpz] ++;
					Nm[document] ++;
					Nk[tmpz] ++;
					z[document][i] = tmpz; //


				}
			}

		}
		
	}
	
	public int gibbsampling(int document, int n){

		Nmk[document][z[document][n]] --; // Nmk --
		Nkt[corpus[document][n]][z[document][n]] --; // Nkt --
		Nm[document]--;
		Nk[z[document][n]] --;


		int tmpz = 0;
		double[] proba = new double[K];
		for(int i=0;i<K;i++){
			proba[i] = (Nkt[corpus[document][n]][i] + beta)/(Nk[i] + V*beta) * (Nmk[document][i] + alpha);///(Nm[document]+ K*alpha);

		}

		for(int i=1;i<K;i++)
			proba[i] += proba[i-1];

		double u = Math.random() * proba[K-1]; //random u
		for(tmpz=0;tmpz<K;tmpz++){
			if(u <= proba[tmpz])
				break;
		}

		return tmpz;






	}

	public void computePhi(){
		/**
		 * compute phi[][]
		 */
		for(int k =0; k< K;k ++){
			for(int t=0;t<V;t++)
				phi[k][t] = (Nkt[t][k] + beta)/(Nk[k] + V*beta);
		}

	}
	public void computeTheta(){

		/**
		 * compute theta[][]
		 */
		for(int m=0;m<M;m++){
			for(int k=0;k<K;k++)
				theta[m][k] = (Nmk[m][k] + alpha)/(Nm[m] + K*alpha);
		}

	}
	
	public void computeLogLikelihood(){
		/**
		 * compute loglikelihood
		 */
		double logllhood = 0.0;
		double sum = 0.0;
		for(int m=0;m<M;m++){
			for(int n=0;n<z[m].length;n++){
				sum = 0.0;
				for(int k=0;k<K;k++){
					sum += theta[m][k] * phi[k][corpus[m][n]];
				}
				logllhood += Math.log(sum);
			}
		}
		long endTime=System.currentTimeMillis(); 
		System.out.println((endTime-startTime)+"ms");
//		String prefix = "_K" + K + "_alpha" + alpha+"_beta" + beta;
//		System.out.println(prefix + "," + logllhood);
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		
		LDAarguments arg = new LDAarguments();
		arg.Setalpha(0.1);arg.Setbeta(0.01);
		arg.Setiterations(500);arg.SetK(50);
		arg.SetM(1500);arg.SetV(12419);
		
		
		LDA lda = new LDA();
		lda.init(arg);
		lda.readFile("nips.txt");
		lda.Estimation();

	}

}
