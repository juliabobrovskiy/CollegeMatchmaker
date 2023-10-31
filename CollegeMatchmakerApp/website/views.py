import datetime
import pickle
from flask import Blueprint, redirect, render_template, request, flash, jsonify, session, url_for
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


user_input_dict = {'Associates':3.0, '2-4 Academic Years':4.0, 'Bachelors':5.0, 'Postbaccalaureate':6.0, 'Masters':7.0, 'Post-Masters':8.0, 
              'Doctors':9.0, 'Under 1000':1.0, '1000–4999':2.0, '5000–9999':3.0, '10000–19999':4.0, '20000 and above':5.0, "Yes":1.0, "No":0.0} 

views = Blueprint('views', __name__)

# Load models
with open('kmeans_knn_model.pkl', 'rb') as file:
    kmeans_loaded, knn_models_loaded = pickle.load(file)

# Load clustered data
# data = pd.read_csv('kmeans_clustered_data.csv')
data = pd.read_csv('merged_clustering.csv')
list_of_colleges = data['inst_name'].values.tolist()




def get_attributes_from_school(identifier):
    # Filter the DataFrame based on the identifier in col1

    filtered_df = data[data['inst_name'] == identifier]

    # Extract columns 2 to 5
    result_df = filtered_df[['offering_highest_level','inst_size','hbcu','medical_degree','tribal_college','land_grant',
                            'inst_affiliation','oncampus_housing','calendar_system','study_abroad','dual_credit','ap_credit',
                            'employment_services','placement_services','oncampus_daycare','disability_indicator']].values.tolist()
    return result_df







@views.route('/input', methods=['GET', 'POST'])
def input():
    if request.method == 'POST': 

        #Schoolinput data
        test_select = request.form.get('selected_option')
        session['test_select'] = test_select #not in use



        #user input daata
        offering_highest_level = request.form.get('offering_highest_level')
        inst_size = request.form.get('inst_size')
        hbcu = request.form.get('hbcu')
        medical_degree = request.form.get('medical_degree')
        tribal_college = request.form.get('tribal_college')
        land_grant = request.form.get('land_grant')
        inst_affiliation = request.form.get('inst_affiliation')
        oncampus_housing = request.form.get('oncampus_housing')
        calendar_system = request.form.get('calendar_system')
        study_abroad = request.form.get('study_abroad')
        dual_credit = request.form.get('dual_credit')
        ap_credit = request.form.get('ap_credit')
        employment_services = request.form.get('employment_services')
        placement_services = request.form.get('placement_services')
        oncampus_daycare = request.form.get('oncampus_daycare')
        disability_indicator = request.form.get('disability_indicator')

        radio_select = request.form.get('exampleRadios')
        session['radio_select'] = radio_select

        user_attributes = ",".join([offering_highest_level, inst_size, hbcu, medical_degree, tribal_college, land_grant, inst_affiliation, oncampus_housing, calendar_system,
                     study_abroad, dual_credit, ap_credit, employment_services, placement_services, oncampus_daycare, disability_indicator])
        session['user_attributes'] = user_attributes

        # print(user_attributes)

        # session['button1'] = request.form.get('action')

        return redirect(url_for('views.predict'))
        
    all_colleges = list_of_colleges
    return render_template('input.html', all_colleges=all_colleges)



@views.route("/predict", methods=['GET'])
def predict():
    print("TEST LINE00")
    #selector = False#test to use Predefined School
    #selector = session.get('selector')
    test_select = session.get('test_select')

    radio_selector = session.get('radio_select')
    # print(radio_selector)
    


    if radio_selector == 'option1':#WORK WITH ATTRIBITES
        # Handle the case where we use user input
        test_select = session.get('user_attributes')
        print("test_select", test_select)
        # Input string
        attr_list = test_select.split(",")

        int_list =[]

        int_list = [user_input_dict[string] for string in attr_list]

        test_select = [int_list]
        

        # Split the string into elements and convert them to integers using map() function
        # int_list = list(map(int, test_select[0].split(",")))

        print(int_list)  # Output: [1, 2, 3, 4, 5]

    
    else:#WORK WITH SCHOOLS
        test_select = get_attributes_from_school(test_select)

    
    print(test_select)
    
    # Assign student to a cluster using K-Means
    cluster_assignments = kmeans_loaded.predict(test_select) ##list or no?new_student_data

    colleges = []
    
    # Find k nearest universities using KNN model for each student
    for idx, cluster in enumerate(cluster_assignments):
        knn_model, original_indices = knn_models_loaded[cluster]
        distances, relative_indices = knn_model.kneighbors([test_select[idx]])#new_student_data

        # relative indices --> original indices
        recommended_indices = [original_indices[i] for i in relative_indices[0]]

        recommended_universities = data.iloc[recommended_indices]

        colleges.append(list(recommended_universities.inst_name))
        #need to convert college names into string for map usage
        string_of_colleges = ','.join(colleges[0])
        print(string_of_colleges)


    return render_template('result.html', predictions = colleges, test_select=test_select, string_of_colleges=string_of_colleges)




@views.route("/map", methods=['GET'])
def map():
    #current_time = datetime.date.today().isoformat()
    return render_template('map.html')

@views.route("/test", methods=['GET'])
def test():
    
    return render_template('test.html')



@views.route("/health", methods=['GET'])
def health():
    current_time = datetime.date.today().isoformat()
    return {"time": current_time}





@views.route('/', methods=['GET'])
def home():
    test = 1234
    return render_template("index.html")


