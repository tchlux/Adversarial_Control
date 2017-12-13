using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using UnityStandardAssets.Vehicles.Car;
using UnityEngine.SceneManagement;
using System.IO;
using System.Collections.Generic;

public class UISystem : MonoSingleton<UISystem> {

    public CarController carController;
    public string GoodCarStatusMessage;
    public string BadSCartatusMessage;
    public Text MPH_Text;
    public Image MPH_Animation;
    public Text Angle_Text;
    public Text RecordStatus_Text;
	public Text DriveStatus_Text;
	public Text SaveStatus_Text;
    public GameObject RecordingPause; 
	public GameObject RecordDisabled;
	public bool isTraining = false;

    private bool recording;
    private float topSpeed;
	private bool saveRecording;
	private Dictionary<int, Transform> carLocations = new Dictionary<int, Transform>();
	private int minFrameNum = 0;


    // Use this for initialization
    void Start() {
		Debug.Log (isTraining);
        topSpeed = carController.MaxSpeed;
        recording = false;
        RecordingPause.SetActive(false);
		RecordStatus_Text.text = "RECORD";
		DriveStatus_Text.text = "";
		SaveStatus_Text.text = "";
		SetAngleValue(0);
        SetMPHValue(0);
		if (!isTraining) {
			DriveStatus_Text.text = "Mode: Autonomous";
			RecordDisabled.SetActive (true);
			RecordStatus_Text.text = "";
		} 
    }

    public void SetAngleValue(float value)
    {
        Angle_Text.text = value.ToString("N2") + "°";
    }

    public void SetMPHValue(float value)
    {
        MPH_Text.text = value.ToString("N2");
        //Do something with value for fill amounts
        MPH_Animation.fillAmount = value/topSpeed;
    }

    public void ToggleRecording()
    {
		// Don't record in autonomous mode
		if (!isTraining) {
			return;
		}

        if (!recording)
        {
			if (carController.checkSaveLocation()) 
			{
				recording = true;
				RecordingPause.SetActive (true);
				RecordStatus_Text.text = "RECORDING";
				carController.IsRecording = true;
			}
        }
        else
        {
			saveRecording = true;
			carController.IsRecording = false;
        }
    }
	
    void UpdateCarValues()
    {
        SetMPHValue(carController.CurrentSpeed);
        SetAngleValue(carController.CurrentSteerAngle);
    }

	// Update is called once per frame
	void Update () {

        // Easier than pressing the actual button :-)
        // Should make recording training data more pleasant.

		if (carController.getSaveStatus ()) {
			SaveStatus_Text.text = "Capturing Data: " + (int)(100 * carController.getSavePercent ()) + "%";
			//Debug.Log ("save percent is: " + carController.getSavePercent ());
		} 
		else if(saveRecording) 
		{
			SaveStatus_Text.text = "";
			recording = false;
			RecordingPause.SetActive(false);
			RecordStatus_Text.text = "RECORD";
			saveRecording = false;
		}

        if (Input.GetKeyDown(KeyCode.R))
        {
            ToggleRecording();
        }

		if (!isTraining) 
		{
			if ((Input.GetKey(KeyCode.W)) || (Input.GetKey(KeyCode.S))) 
			{
				DriveStatus_Text.color = Color.red;
				DriveStatus_Text.text = "Mode: Manual";
			} 
			else 
			{
				DriveStatus_Text.color = Color.white;
				DriveStatus_Text.text = "Mode: Autonomous";
			}
		}

	    if(Input.GetKeyDown(KeyCode.Escape))
        {
            //Do Menu Here
            SceneManager.LoadScene("MenuScene");
        }

        if (Input.GetKeyDown(KeyCode.Return))
        {
            //RESET CAR POSITION TO BEGINNING
			SceneManager.LoadScene(Application.loadedLevelName);
			// ResetToFrame(Time.frameCount - 10);
        }

        UpdateCarValues();
	UpdateCarPositions();
	ResetCarFromFile();
    }
		
	static void ResetCarFromFile()
	{
		string path = "Assets/1_SelfDrivingCar/Text/reset.txt";

		//Read the text from directly from the reset.txt file
		StreamReader reader = new StreamReader(path); 
		if (reader.ReadToEnd ().Length > 0) {
			reader.Close();
			SceneManager.LoadScene(Application.loadedLevelName);
			StreamWriter writer = new StreamWriter(new FileStream(path, FileMode.Truncate));
			writer.Write("");
			writer.Close();
		}
		reader.Close();
	}


	// Model makes predictions AND Image current view of car
	// Image goes into model, predicts right steering angle, angle goes back in
	// INSTEAD: Perturbed image goes into the model, predict different angle
	void UpdateCarPositions()
	{
		string path = "Assets/1_SelfDrivingCar/Text/carlocation.txt";

		// Read the text from directly from the carlocation.txt file
		StreamWriter writer = new StreamWriter(new FileStream(path, FileMode.Truncate));
		Transform t = carController.transform;
		writer.Write(t.position.x + " " + t.position.y + " " + t.position.z + " " + Time.frameCount);
		writer.Close();

		if (carLocations.Count == 100) {
			carLocations.Remove(minFrameNum);
		}
		carLocations.Add (Time.frameCount, t);
		minFrameNum = Time.frameCount - 99;
	}

	void ResetToFrame(int frameNum) {
		Transform carLocationAtFrame = carLocations [frameNum];
		carController.transform.position = carLocationAtFrame.position;
		carController.transform.rotation = carLocationAtFrame.rotation;
		carController.transform.localScale = carLocationAtFrame.localScale;
	}
}
