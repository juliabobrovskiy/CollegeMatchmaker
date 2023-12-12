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


# Assuming you've saved the scaler as 'scaler.pkl'
with open('scaler.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)
# Load clustered data
# data = pd.read_csv('kmeans_clustered_data.csv')
# data = pd.read_csv('merged_clustering.csv')
unitid_map = pd.read_csv('unitid_mapping.csv')
data = pd.read_csv('merged_df_cleaned.csv')
model_data = pd.read_csv('kmeans_clustered_data.csv')


list_of_colleges = data['inst_name'].values.tolist()
list_of_colleges = list(set(list_of_colleges))
list_of_colleges = sorted(list_of_colleges)

#NEW DATA
#unitid,offering_highest_level,offering_highest_degree,offering_grad,inst_size,hbcu,medical_degree,tribal_college,land_grant,sector,inst_control,fips,inst_affiliation,oncampus_housing,calendar_system,study_abroad,dual_credit,ap_credit,employment_services,placement_services,oncampus_daycare,disability_indicator,disability_percentage,cont_prof_prog_offered,occupational_prog_offered,acceptance_rate,tuition_fees_ft,cluster

unitid = 12345

def get_attributes_from_school(identifier):
    # Filter the DataFrame based on the identifier in col1

    unitid = data[data['inst_name'] == identifier].unitid
    print("get_attributes name and ID", identifier, unitid)

    filtered_df = model_data[model_data['unitid'] == int(unitid)]

    # Extract columns 2 to 5

    result_df = filtered_df[['offering_highest_level','offering_highest_degree','offering_grad','inst_size','hbcu','medical_degree',
                             'tribal_college','land_grant','sector','inst_control','fips','inst_affiliation','oncampus_housing','calendar_system',
                             'study_abroad','dual_credit','ap_credit','employment_services','placement_services','oncampus_daycare','disability_indicator',
                             'disability_percentage','cont_prof_prog_offered','occupational_prog_offered','acceptance_rate','tuition_fees_ft']].values.tolist()



    # result_df = filtered_df[['offering_highest_level','inst_size_x','hbcu','medical_degree','tribal_college','land_grant',
    #                         'inst_affiliation','oncampus_housing','calendar_system','study_abroad','dual_credit','ap_credit',
    #                         'employment_services','placement_services','oncampus_daycare','disability_indicator']].values.tolist()
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
        ###############################################
        offering_highest_degree = request.form.get('offering_highest_degree')
        offering_grad = request.form.get('offering_grad')
        sector = request.form.get('sector')
        inst_control = request.form.get('inst_control')
        fips = request.form.get('fips')
        disability_percentage = request.form.get('disability_percentage')
        cont_prof_prog_offered = request.form.get('cont_prof_prog_offered')
        occupational_prog_offered = request.form.get('occupational_prog_offered')
        acceptance_rate = request.form.get('acceptance_rate')
        tuition_fees_ft = request.form.get('tuition_fees_ft')

        radio_select = request.form.get('exampleRadios')
        session['radio_select'] = radio_select

        # user_attributes = ",".join([offering_highest_level, inst_size, hbcu, medical_degree, tribal_college, land_grant, inst_affiliation, oncampus_housing, calendar_system,
        #              study_abroad, dual_credit, ap_credit, employment_services, placement_services, oncampus_daycare, disability_indicator])
        

        user_attributes = ",".join([offering_highest_level,offering_highest_degree,offering_grad,inst_size,hbcu,medical_degree,tribal_college,
                                    land_grant,sector,inst_control,fips,inst_affiliation,oncampus_housing,calendar_system,study_abroad,dual_credit,
                                    ap_credit,employment_services,placement_services,oncampus_daycare,disability_indicator,disability_percentage,
                                    cont_prof_prog_offered,occupational_prog_offered,acceptance_rate,tuition_fees_ft])


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
    inst_name = session.get('test_select')

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

        # print(int_list)  # Output: [1, 2, 3, 4, 5]

    
    else:#WORK WITH SCHOOLS


        test_select = get_attributes_from_school(inst_name)

    
    # print(test_select)
    


    new_student_data_scaled = scaler_loaded.transform(test_select)
    # Assign student to a cluster using K-Means
    cluster_assignments = kmeans_loaded.predict(new_student_data_scaled) ##list or no?new_student_data

    colleges = []
  
    



    # Find k nearest universities using KNN model for each student
    for idx, cluster in enumerate(cluster_assignments):

        knn_model, original_indices, unitids = knn_models_loaded[cluster]
        distances, indices = knn_model.kneighbors([new_student_data_scaled[idx]])

        recommended_indices = unitids.iloc[indices[0][1:]].values
        # print(recommended_indices)

        for index in recommended_indices:

            recommended_university = data[data['unitid']==index]
            # print(recommended_university)
            colleges.append(list(recommended_university.inst_name))

    # print(colleges)
    colleges = [college[0] for college in colleges]
    string_of_colleges = ','.join(colleges)

    recommended_universities = data[data['inst_name'].isin(colleges)].copy()
    print(recommended_universities)



    unitid = data[data['inst_name'] == inst_name].unitid
    unitid = str(int(unitid))
    
    
    unitid = "UnitID_" + unitid + "_Predicted_Class_7_SHAP_Waterfall_Plot.png"
    print("looking for inutid", unitid)
    
    return render_template('result.html', predictions = colleges, test_select=test_select, string_of_colleges=string_of_colleges, recommended_universities=recommended_universities, unitid=unitid)




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


