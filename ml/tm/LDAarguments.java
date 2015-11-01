package ml.tm;

public class LDAarguments {
	
	public double alpha;
	public double beta;
	public int K;
	public int V;
	public int M;
	public int iterations;
	public int savestep;
	
	public void Setalpha(double alpha){
		this.alpha = alpha;
	}
	
	public void Setbeta(double beta){
		this.beta = beta;
	}
	
	public void SetK(int K){
		this.K = K;
	}
	
	public void SetV(int V){
		this.V = V;
	}
	
	public void SetM(int M){
		this.M = M;
	}
	
	public void Setiterations(int iterations){
		this.iterations = iterations;
	}
	
	public void Setsavestep(int savestep){
		this.savestep = savestep;
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
