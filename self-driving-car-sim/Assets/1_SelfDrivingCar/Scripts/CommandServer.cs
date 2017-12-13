using UnityEngine;
using System.Collections.Generic;
using System.Collections;
using SocketIO;
using UnityStandardAssets.Vehicles.Car;
using System;
using System.Security.AccessControl;
using System.IO;
using System.Collections.Generic;
using UnityEngine.UI;

public class CommandServer : MonoBehaviour
{
	public CarRemoteControl CarRemoteControl;
	public Camera FrontFacingCamera;
	private SocketIOComponent _socket;
	private CarController _carController;

	private Image canvas;
	private float oldSteeringAngle;
	private float oldAcceleration;
	private float oldFrameNum;

	private Color clear;
	private Color full;

	// Use this for initialization
	void Start()
	{
		_socket = GameObject.Find("SocketIO").GetComponent<SocketIOComponent>();
		_socket.On("open", OnOpen);
		_socket.On("steer", OnSteer);
		_socket.On("manual", onManual);
		_carController = CarRemoteControl.GetComponent<CarController>();

		canvas = GameObject.FindWithTag ("Adversarial").GetComponent<Image>();
		clear = canvas.color;
		full = canvas.color;
		clear.a = 0.0f;
		full.a = 1.0f;
		canvas.color = full;
	}

	void Awake()
	{
	//	Application.targetFrameRate = ;
	//      Debug.Log (1.0f / Time.deltaTime);
	}

	// Update is called once per frame
	void Update()
	{
		Debug.Log ("UDPATE (FR):" + (1.0f / Time.deltaTime));
	}

	void OnOpen(SocketIOEvent obj)
	{
		Debug.Log("Connection Open");
		EmitTelemetry(obj);
	}

	// 
	void onManual(SocketIOEvent obj)
	{
		EmitTelemetry (obj);
	}


	void OnSteer(SocketIOEvent obj)
	{
		JSONObject jsonObject = obj.data;
		
		// Canvas not on, set the value in the car, 
		if (canvas.color.a == 0.0f) {
			Debug.Log ("canvas off");
			string path = "Assets/1_SelfDrivingCar/Text/carsteer.txt";

			// Read the text from directly from the carsteer.txt file
			// Debug.Log (oldFrameNum + " " + Time.frameCount);
			float stAng = float.Parse(jsonObject.GetField("steering_angle").str);
			float acc = float.Parse(jsonObject.GetField("throttle").str);
			StreamWriter writer = new StreamWriter(new FileStream(path, FileMode.Truncate));
			writer.Write(oldSteeringAngle + " " + oldAcceleration + " " + stAng + " " + acc + " " + Time.frameCount);
			writer.Close();
			CarRemoteControl.SteeringAngle = stAng;
			CarRemoteControl.Acceleration = acc;
			canvas.color = full;
		}
		// The canvas is on, perturbed image available
		else {
			Debug.Log ("canvas on");

			oldSteeringAngle = float.Parse(jsonObject.GetField("steering_angle").str);
			oldAcceleration = float.Parse(jsonObject.GetField("throttle").str);
			oldFrameNum = Time.frameCount;
			canvas.color = clear;
		}
		EmitTelemetry (obj);
	}

	void EmitTelemetry(SocketIOEvent obj)
	{
		UnityMainThreadDispatcher.Instance().Enqueue(() =>
		{
			// print("Attempting to Send...");
			// send only if it's not being manually driven
			if ((Input.GetKey(KeyCode.W)) || (Input.GetKey(KeyCode.S))) {
				_socket.Emit("telemetry", new JSONObject());
			}
			else {
				// Collect Data from the Car
				Dictionary<string, string> data = new Dictionary<string, string>();
				data["steering_angle"] = _carController.CurrentSteerAngle.ToString("N4");
				data["throttle"] = _carController.AccelInput.ToString("N4");
				data["speed"] = _carController.CurrentSpeed.ToString("N4");
				data["image"] = Convert.ToBase64String(CameraHelper.CaptureFrame(FrontFacingCamera));
				_socket.Emit("telemetry", new JSONObject(data));
			}
			// Must sleep to let model come up with other perturbation quick enough
		});

		//    UnityMainThreadDispatcher.Instance().Enqueue(() =>
		//    {
		//      	
		//      
		//
		//		// send only if it's not being manually driven
		//		if ((Input.GetKey(KeyCode.W)) || (Input.GetKey(KeyCode.S))) {
		//			_socket.Emit("telemetry", new JSONObject());
		//		}
		//		else {
		//			// Collect Data from the Car
		//			Dictionary<string, string> data = new Dictionary<string, string>();
		//			data["steering_angle"] = _carController.CurrentSteerAngle.ToString("N4");
		//			data["throttle"] = _carController.AccelInput.ToString("N4");
		//			data["speed"] = _carController.CurrentSpeed.ToString("N4");
		//			data["image"] = Convert.ToBase64String(CameraHelper.CaptureFrame(FrontFacingCamera));
		//			_socket.Emit("telemetry", new JSONObject(data));
		//		}
		//      
		////      
		//    });
	}
}