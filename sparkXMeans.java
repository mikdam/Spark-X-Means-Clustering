
package sparkML;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.storage.StorageLevel;

public class sparkXMeans {
    
    // Finding the right k by splitting individual clusters into two and
    // comparing the BIC of the parent cluster to that of childern clusters.
    public static KMeansModel Clustering(JavaRDD<Vector> V, int kMin, int kMax){
       KMeansModel clusters;
       
       
       long R, M, R1, R2;
       int newk, k = kMin;
       double BIC1, BIC2;
       
       V.persist(StorageLevel.MEMORY_ONLY());
       
       while (k < kMax){
           newk = 0;
           
           clusters = KMeans.train(V.rdd(), k, 1000000, 10);
           JavaPairRDD<Vector,Integer> labelled_data = V.zip(clusters.predict(V));
           labelled_data.persist(StorageLevel.MEMORY_ONLY());
           
           for(int i=0; i<k; i++){
               final int x = i;  
               
               JavaRDD<Vector> Points1 = labelled_data.filter(a -> a._2 == x).map(a-> a._1); 
               Points1.persist(StorageLevel.MEMORY_ONLY());
               R = Points1.count();
               
               
               if((R>0)){
                   KMeansModel test_clusters = KMeans.train(Points1.rdd(), 2, 10000);
                   JavaPairRDD<Vector,Integer> Points2 = Points1.zip(test_clusters.predict(Points1));
                   Points2.persist(StorageLevel.MEMORY_ONLY());


                   M = clusters.clusterCenters()[i].size();
                   R1 = Points2.filter(a -> a._2 == 0).count();
                   R2 = Points2.filter(a -> a._2 == 1).count();
                   
                   if (R1!=0 && R2!=0) {
                       
                       BIC1 = CalculateBIC(clusters.clusterCenters()[i], Points1, R, M);
                       BIC2 = CalculateBIC(test_clusters.clusterCenters(), Points2, R, M);
                       
                       if (BIC1 > BIC2) 
                           newk++;
                       else 
                           newk += 2;                       
                   
                   } else
                       newk++;
                   
                   Points2.unpersist();
               }
               
               Points1.unpersist();
               
               if(newk >= kMax){
                   newk  = kMax;
                   break;
               }
           }
           
           if(k == newk)
               break;
           else
               k = newk;    

       }
       
       clusters = KMeans.train(V.rdd(), k, 100000,10);
       V.unpersist();
       
       return clusters;
    }
    
    // Finding the right k by calculating BIC for a range of k values and choosing the one
    // the highest value.
    
    public static KMeansModel Clustering_MaxBIC(JavaRDD<Vector> V, int kMin, int kMax){
       KMeansModel model, bestModel;
       
       
       long R, M;
       int k;
       double BIC, MaxBIC;
       
              
       V.persist(StorageLevel.MEMORY_ONLY());
       
       bestModel = KMeans.train(V.rdd(), kMin, 100000, 10);
       JavaPairRDD<Vector,Integer> labelled_data = V.zip(bestModel.predict(V));
       labelled_data.persist(StorageLevel.MEMORY_ONLY());
       R = labelled_data.count();
       M = bestModel.clusterCenters()[0].size();

       MaxBIC = CalculateBIC(bestModel.clusterCenters(), labelled_data, R, M);

       labelled_data.unpersist();
       
       for(k = kMin+1; k <= kMax; k++){           
           model = KMeans.train(V.rdd(), k, 100000, 10);
           labelled_data = V.zip(model.predict(V));
           labelled_data.persist(StorageLevel.MEMORY_ONLY());
           
           BIC = CalculateBIC(model.clusterCenters(), labelled_data, R, M);
           
           labelled_data.unpersist();
           
           if(BIC > MaxBIC){
               MaxBIC = BIC;
               bestModel = model;
           }
       }
       
       V.unpersist();
             
       return bestModel;
    }
    
    // Calculating BIC for a single cluster (parent, usually)
    public static double CalculateBIC(Vector C, JavaRDD<Vector> Data, long R, long M){
        double bic,l;
        double K=1, Rn, sigma2;
        
        Rn = R;
        
        sigma2 = ClusterVariance(C, Data,R, M);
        
        l =    Rn * Math.log(Rn) 
             - Rn *  Math.log(R)
             - ((M*Rn) / 2.0) *  Math.log(2.0 * Math.PI * sigma2 )
             - M*(Rn - 1.0) / 2.0; 
        
        bic = l - (K * (M+1.0))/ 2.0 * Math.log(R);
        
        return bic;
    }
    
    // Calculating BIC for a set of clusters
    public static double CalculateBIC(Vector[] C, JavaPairRDD<Vector,Integer> LabelledData, long R, long M){
        double bic, l=0, sigma2;
        long  Rn, K; 
                
        K = C.length;
        sigma2 = ClusterVariance(C, LabelledData,R, M,K); 
        
        for(int i=0; i<K; i++){
            final int x = i;
            Rn = LabelledData.filter(a -> a._2 == x).count();
          
            l +=  Rn * Math.log(Rn) 
                - Rn *  Math.log(R)
                - ((M*Rn) / 2.0) *  Math.log(2.0 * Math.PI * sigma2)
                - M*(Rn - 1.0) / 2.0;        
        }
        
        bic = l - (K * (M+1.0))/2.0 * Math.log(R);
        
        return bic;
    }
    
    //Calculating variance for one cluster
    public static double ClusterVariance(Vector C, JavaRDD<Vector> Data, long R, long M) {
        double v;
        final Vector V = C;
        
        v = Data.map(p -> Vectors.sqdist(p, V) )
                .reduce((s1,s2) -> s1+s2) / (M*(R-1));
        
        return v;
    }
    
    // Calculating variance for a set of clusters
    public static double ClusterVariance(Vector C[], JavaPairRDD<Vector,Integer> LabelledData, long R, long M, long K) {
        double v, sum = 0;
        
        for(int i=0; i<K; i++){
            final int x = i;
            final Vector V = C[i];
            
            sum += LabelledData.filter(a -> a._2 == x)
                    .map(a -> Vectors.sqdist(a._1, V) )
                    .reduce((s1,s2) -> s1+s2);
        }
        
        v = sum / (M*(R-K));
        return v;
    }
    
}
