import "./App.css";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faMessage, faPaperPlane } from "@fortawesome/free-solid-svg-icons";
import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { v4 as uuidv4 } from "uuid";

const BotMessageBubble = ({ message }) => {
  return (
    <div>
      <div className="ml-5 font-medium text-black">Assistant</div>
      <pre className="bg-[#334155] text-white m-3 font-sans rounded-t-3xl rounded-br-3xl p-3 text-wrap shadow-lg max-w-[70%] bubble">
        <p>{message}</p>
      </pre>
    </div>
  );
};

const UserMessageBubble = ({ message }) => {
  return (
    <div className="max-w-[70%] ml-auto">
      <div className="m-2">User</div>
      <p className="bg-[#6b7280] text-white m-2 rounded-t-2xl rounded-bl-2xl p-3 shadow-lg bubble">{message}</p>
    </div>
  );
};

const LoadingBubble = () => {
  return (
    <div>
      <div className="ml-3 font-medium text-black">Assistant</div>
      <pre className="bg-[#334155] text-white m-3 font-sans rounded-t-3xl rounded-br-3xl p-3 text-wrap shadow-lg max-w-[70%] bubble">
        <p className="loading-bubble bubble">Loading...</p>
      </pre>
    </div>
  );
};

function App() {
  const [visible, setVisible] = useState(false);
  const [messages, setMessages] = useState([]);
  const [userId, setUserId] = useState(""); 
  const [input, setInput] = useState(''); 
  const [disableInput, setDisableInput] = useState(false); 
  const messageEndRef = useRef(null); 

  useEffect(() => {
    const generatedUserId = uuidv4();
    setUserId(generatedUserId);
  }, []);

  useEffect(() => {
    if (messageEndRef.current) {
      messageEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (input.trim() === "") return;
  
    setInput('');
    setMessages((prevMessages) => [
      ...prevMessages,
      {
        user: "user",
        type: "message",
        message: input,
      },
      {
        user: "bot",
        type: "loading",
        message: "Loading",
      },
    ]);
  
    try {
      setDisableInput(true);
  
      const response = await axios.post(
        "http://127.0.0.1:5000/form", // Change this to your actual backend URL
        {
          query: input,
          user_id: userId,
        }
      );
  
      if (response.status === 200) {
        const botResponse = response.data.response;
  
        let formattedMessage;
  
        if (typeof botResponse === "object") {
          // Format the object response into a readable string
          formattedMessage = Object.entries(botResponse)
            .map(([key, value]) => `${key}: ${value}`)
            .join("\n");
        } else {
          // If it's just a string, display it directly
          formattedMessage = botResponse;
        }
  
        setMessages((prevMessages) => [
          ...prevMessages.filter((msg) => msg.type !== "loading"),
          { user: "bot", type: "message", message: formattedMessage },
        ]);
      } else {
        setMessages((prevMessages) => [
          ...prevMessages.filter((msg) => msg.type !== "loading"),
          { user: "bot", type: "message", message: "Some error has occurred." },
        ]);
      }
    } catch (error) {
      console.error(error);
      setMessages((prevMessages) => [
        ...prevMessages.filter((msg) => msg.type !== "loading"),
        { user: "bot", type: "message", message: "An error occurred. Please try again later." },
      ]);
    } finally {
      setDisableInput(false);
    }
  };
  

  return (
    <>
      <button
        className="absolute right-2 bottom-4 m-2 bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-500 hover:to-gray-600 rounded-full w-14 h-14 flex items-center justify-center shadow-lg"
        onClick={() => setVisible((prevVisible) => !prevVisible)}
      >
        <FontAwesomeIcon icon={faMessage} color="white" />
      </button>
      
      {visible && (
        <div className="absolute bg-gradient-to-b from-[#1a003a] to-black sm:right-[2vw] sm:bottom-[12vh] h-[80vh] w-[25vw] rounded-xl bottom-[11vh] right-4 shadow-2xl border border-gray-700">
          <div className="h-[8vh] bg-[#0f172a] rounded-t-xl flex items-center justify-center shadow-md">
            <div className="text-white text-2xl font-bold">Energy Management</div>
          </div>
          
          <div className="h-[72vh] flex-grow rounded-b-xl grid bg-white">
            <div className="overflow-y-auto h-[65vh] flex flex-col p-2 space-y-2">
              {messages.map((item, index) => {
                if (item.user === "user") {
                  return <UserMessageBubble message={item.message} key={index} />;
                } else if (item.type === "loading") {
                  return <LoadingBubble key={index} />;
                } else {
                  return <BotMessageBubble message={item.message} key={index} />;
                }
              })}
              <div ref={messageEndRef}></div>
            </div>

            <form onSubmit={sendMessage} className="flex flex-row items-center p-2">
              <input
                onChange={(e) => setInput(e.target.value)}
                type="text"
                disabled={disableInput}
                value={input}
                placeholder="Enter your message"
                className="p-2 flex-grow rounded-lg border border-gray-600 bg-gray-700 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-black"
              />
              <button
                type="submit"
                className="ml-2 rounded-lg w-10 h-10 flex items-center justify-center hover:bg-[#94a3b8] bg-[#3f3f46] text-[#d4d4d8] shadow-lg transition-colors duration-300 ease-in-out"
              >
                <FontAwesomeIcon icon={faPaperPlane} color="#d4d4d8" />
              </button>
            </form>
          </div>
        </div>
      )}
    </>
  );
}

export default App;
