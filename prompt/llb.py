from prompt import PromptingMethod
import re

system_prompt = """
You are StanWriter. Given a description of a problem, you write Stan code.
You always first write THOUGHTS START and then describe your model in words then THOUGHTS END. Then you write MODEL START and write the Stan code then MODEL END.
Be creative when coming up with your model!
When you declare arrays, ALWAYS use the syntax "array[2,5] int x" NEVER the old syntax "int x[2,5]".
In the generated quantities block ALWAYS use the _rng syntax like "x ~ normal_rng(mu,sigma)" NEVER "x ~ normal(mu,
sigma)".
NEVER write ‘‘‘stan before your code.
ALWAYS give priors for all variables. NEVER use implicit/improper priors.
NEVER use "target += ..." syntax.
"""

exemplars = [
    """
    PROBLEM
    Given the weight of a bunch of students, infer true mean of all students. Use reasonably informative priors.
    DATA
    int num_students; // number of students
    array[num_students] real student_weight; // weight of each student in kilograms
    GOAL
    real weight_mean; // true mean weight for all students
    
    THOUGHTS START
    I will model this problem by creating two latent variables. ‘weight_mean‘ will reflect the true mean weight of all students, and ‘weight_sigma‘ will reflect the standard deviation. Since I was instructed to use reasonably informative priors, I’ll assume a normal prior for ‘weight_mean‘ with a mean of 80kg and a standard deviation of 20 kg. For the ‘weight_sigma‘, I’ll choose a lognormal prior with mu=log(10) and sigma=log(20). This reflects that my best guess for the standard deviation is around 10 kg, but could be significantly larger or smaller.
    THOUGHTS END
    MODEL START
    data {
        int num_students;
        array[num_students] real student_weight;
    }
    parameters {
        real weight_mean;
        real<lower=0> weight_sigma;
    }
    model {
        weight_mean ~ normal(80,20);
        weight_sigma ~ lognormal(log(10),log(20));
        for (i in 1:num_students){
            student_weight[i] ~ normal(weight_mean, weight_sigma);
        }
    }
    MODEL END
    """,
    """
    PROBLEM
    Given a bunch of binary features of a movie, predict if a user will like it or not.
    DATA
    int num_train; // number of movies to train on
    int num_test; // number of movies to test on
    int num_features; // number of binary features for each movie
    array[num_train, num_features] int train_features; // binary features
    array[num_train] int train_like; // did the user like the movie
    array[num_train, num_features] int test_features; // binary features
    GOAL
    array[num_test] int test_like; // does the user like each movie in the test set?
    
    THOUGHTS START
    I will model this problem by creating a vector ‘beta‘ of size ‘num_features‘. When the inner-product of this vector is taken with the features of a movie, it gives the logit for how likely the user is to like that movie. I’ll sample each entry of that vector from a standard normal, which seems like a reasonable scale when being combined with binary features. I’ll then sample both ‘train_like‘ and ‘test_like‘ by taking the inner-product of ‘beta‘ with each row of ‘train_features‘ and ‘test_features‘, respectively.
    THOUGHTS END
    MODEL START
    data {
        int num_train;
        int num_test;
        int num_features;
        array[num_train, num_features] int<lower=0, upper=1> train_features;
        array[num_train] int<lower=0, upper=1> train_like;
        array[num_test, num_features] int<lower=0, upper=1> test_features;
    }
    parameters {
        vector[num_features] beta;
    }
    model {
        beta ~ normal(0,1);
        for (n in 1:num_train){
            train_like[n] ~ bernoulli_logit(to_row_vector(train_features[n]) * beta);
        }
    }
    generated quantities {
        array[num_test] int<lower=0, upper=1> test_like;
        for (n in 1:num_test){
            test_like[n] = bernoulli_logit_rng(to_row_vector(test_features[n]) * beta);
        }
    }
    MODEL END
    """,
    """
    PROBLEM
    Given a bunch of like ratings of various users of various movies, predict if users will like future movies. Do this by inferring underlying features of users and movies.
    DATA
    int num_users; // number of users
    int num_movies; // number of movies
    int num_ratings; // number of observed ratings
    array[num_ratings] int user; // what user did rating
    array[num_ratings] int movie; // what movie was rated
    array[num_train, num_features] int like; // did user like the movie (1 if yes, 0 if no)
    GOAL
    int array[num_users, num_movies] all_ratings; // would each user like each movie?
    
    THOUGHTS START
    I will model this problem by assuming that both users and movies can be described in terms of ‘num_features‘=10 features. I’ll create a latent variable ‘user_features‘ that describes the features for each user and a latent variable ‘movie_features‘ that describes the features for each movie. I will assume standard cauchy priors for all these features, indicating that most features are small, but some might be quite large. Finally, I assume that the probability of a given user liking a given movie is given by a bernoulli_logit distribution with a score consisting of the inner-product of the user and movie features. To create the desired output ‘all_ratings‘, I will create a generating quantities block in which I loop over all pairs of users and movies.
    THOUGHTS END
    MODEL START
    data {
        int num_users; // number of users
        int num_movies; // number of movies
        int num_ratings; // number of observed ratings
        array[num_ratings] int user; // what user did rating
        array[num_ratings] int movie; // what movie was rated
        array[num_ratings] int like; // did user like the movie (1 if yes, 0 if no)
    }
    transformed data{
        int num_features = 10;
    }
    parameters {
        array[num_users, num_features] real user_features;
        array[num_movies, num_features] real movie_features;
    }
    model {
        for(k in 1:num_features){
            for(i in 1:num_users){
                user_features[i,k] ~ cauchy(0,1);
            }
            for(j in 1:num_movies){
                movie_features[j,k] ~ cauchy(0,1);
            }
        }
        for (n in 1:num_ratings){
            array[num_features] real x = user_features[user[n]];
            array[num_features] real y = movie_features[movie[n]];
            real score = dot_product(x,y);
            like[n] ~ bernoulli_logit(score);
        }
    }
    generated quantities {
        array[num_users, num_movies] int all_ratings;
        for(i in 1:num_users){
            for(j in 1:num_movies){
                array[num_features] real x = user_features[i];
                array[num_features] real y = movie_features[j];
                real score = dot_product(x,y);
                all_ratings[i,j] = bernoulli_logit_rng(score);
            }
        }
    }
    MODEL END
    """,
    """
    PROBLEM
    There is data for a bunch of different vaccine trials. There are different types of vaccines. Given this data, infer the mean response rate for each trial type. Assume that different vaccines of the same type tend to have similar response rates and that different types of vaccines also usually have response rates that aren’t that different.
    DATA
    int num_types; // number of different types of vaccines
    int num_trials; // number of trials
    array[num_trials] int type; // what type of vaccine tested in each trial
    array[num_trials] int num_subjects; // number of subjects in each trial
    array[num_trials] int responders; // number of subjects in each trial that responded
    GOAL
    array[num_types] response_rate; // mean response rate for a new vaccine of each type
    
    THOUGHTS START
    I’ll create a hierarchical model. The idea is that each trial has some response rate that is drawn from a per-trialtype distribution. Then the parameters for the per-trial-type distributions are all drawn from some global distribution. By sampling the three levels at once (individual trial response rates, per-trial-type response rates, and global rates) all information can be optimally shared. In more detail, I’ll first create parameters ‘a‘ and ‘b‘ for each trial type. These determine the distribution over true response rates for different trials of each type. The true response rate for trial ‘i‘, ‘per_trial_response_rate[i]‘, is drawn from a beta distribution with parameters ‘a[type[i]]‘ and ‘b[type[i]]‘. Then the (observed) number of responders in each trial will be drawn from a binomial distribution with ‘num_subjects[i]‘ repeats and a ‘per_trial_response_rate[i]‘ success rate. To create a prior over ‘a‘ and ‘b‘, I will draw a 2-dimensional vector ‘z[n]‘ for each trial type from some multi normal distribution parameterized by ‘mu‘ and ‘L‘. This will be mapped to ‘a‘ and ‘b‘ in the transformed parameters block through an inverse logit transformation. I’ll sample ‘mu‘ from a broad multivariate normal distribution and ‘L‘ from the distribution that results from sampling from a broad inverse Wishart distribution and then computing the Cholesky factorization.
    THOUGHTS END
    MODEL START
    data {
        int num_types;
        int num_trials;
        array[num_trials] int type;
        array[num_trials] int num_subjects;
        array[num_trials] int responders;
    }
    parameters {
        cholesky_factor_cov[2] L;
        vector[2] mu;
        array[num_types] vector[2] z; // parameters for response rate dist for each type
        array[num_trials] real<lower=0, upper=1> per_trial_response_rate;
    }
    transformed parameters{
        // map unconstrained parameters for each dist type to pair on 0-1 interval
        array[num_types] real a;
        array[num_types] real b;
        array[num_types] real response_rate;
        for(n in 1:num_types){
            a[n] = inv_logit(z[n,1]);
            b[n] = inv_logit(z[n,2]);
            response_rate[n] = a[n]/(a[n]+b[n]); // mean of beta distribution is a/(a+b)
        }
    }
    model {
        mu ~ multi_normal([0,0],[[5,0],[0,5]]);
        L ~ inv_wishart_cholesky(3.0, [[1,0],[0,1]]);
        for(n in 1:num_types){
            z[n] ~ multi_normal_cholesky(mu, L);
        }
        for(i in 1:num_trials){
            per_trial_response_rate[i] ~ beta(a[type[i]], b[type[i]]);
            responders[i] ~ binomial(num_subjects[i], per_trial_response_rate[i]);
        }
    }
    MODEL END
    """,
    """
    PROBLEM
    I start with 1 kg of water. Every day, some fraction of the water (between 0 and 1) evaporates, and I measure how much is remaining. My measurements have a noise of something around 10 g. Predict how much water evaporates every day. It could be anything between 0 and 1, but I suspect it’s something around 0.25. The first day with 1kg isn’t measured.
    DATA
    int num_days;
    array[num_days] measurements; // measurements in kg
    GOAL
    real evaporation; // what fraction evaporates each day?
    
    THOUGHTS START
    I will model this problem by creating a latent variable ‘evaporation‘ which measures what fraction of the water disappears each day. I will sample this from a beta distribution with alpha=1 and beta=3 meaning a mean of around 0.25. Given the evaporation, I will use a for loop to define an array ‘current_water‘ which is the true (unobserved) amount of water remaining after each day. Finally, I will sample ‘measurements[i]‘ for each day i from a normal distribution with a mean of water[i] and a standard deviation of 0.01, chosen because 0.01 kg = 10g.
    THOUGHTS END
    MODEL START
    data {
        int num_days;
        array[num_days] real measurements;
    }
    parameters {
        real<lower=0, upper=1> evaporation;
    }
    transformed parameters{
        array[num_days] real<lower=0, upper=1> water;
        real current_water = 1.0;
        for(i in 1:num_days){
            current_water = current_water * (1-evaporation);
            water[i] = current_water;
        }
    }
    model {
        evaporation ~ beta(1,3);
        for(i in 1:num_days){
            measurements[i] ~ normal(water[i], 0.01);
        }
    }
    MODEL END
    """
]


class LLB(PromptingMethod):
    def __init__(self):
        self.system_prompt = system_prompt
        self.exemplars = exemplars
        self.expected_response_blocks = ['stan_model']

    def build_prompt(self, inst: str, **kwargs):
        _tmp = ''
        for e in self.exemplars:
            _tmp = _tmp + e + '\n\n'
        return self.system_prompt + '\n\nExamples:\n\n' + _tmp + inst

    def parse_response(self, response: str, **kwargs):
        assert isinstance(response, str)
        status = True
        parsed_blocks = {}
        for block in self.expected_response_blocks:
            if block == 'stan_model':
                pattern = r"MODEL START(.*?)MODEL END"
                matches = re.findall(pattern, response, re.DOTALL)
                _tmp = [match.strip() for match in matches]
                if _tmp:
                    parsed_blocks[block] = _tmp[0]
                else:
                    parsed_blocks[block] = self.handle_none_match()
                    status = False
            else:
                pass
        return parsed_blocks, status

    def handle_none_match(self, **kwargs):
        return ''
