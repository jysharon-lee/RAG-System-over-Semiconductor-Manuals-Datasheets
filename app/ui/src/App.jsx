import React, { useState, useRef, useEffect } from 'react';
import { Send, Cpu } from 'lucide-react';
import Citation from './components/Citation';
import './index.css';

function App() {
  const [messages, setMessages] = useState([
    { role: 'bot', content: 'Hello! I am your Datasheet Assistant. What semiconductor spec can I help you find today?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const query = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: query }]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query, top_k: 5 }),
      });

      if (!response.ok) {
        throw new Error('API Error');
      }

      const data = await response.json();
      setMessages(prev => [...prev, { role: 'bot', content: data.answer, sources: data.sources }]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: 'bot', content: 'Sorry, there was an error connecting to the API.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>
          <Cpu className="inline-block mr-3 mb-1" size={36} color="var(--accent-1)" />
          Datasheet RAG
        </h1>
        <p>Intelligent Search across Semiconductor Manuals</p>
      </div>

      <div className="chat-area">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <div className="message-bubble">
              <div className="answer-text">{msg.content}</div>
              
              {msg.sources && msg.sources.length > 0 && (
                <div className="citations-container">
                  <div className="citations-title">Sources</div>
                  <div className="citations-grid">
                    {msg.sources.map((src, i) => (
                      <Citation key={i} source={src} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message bot">
            <div className="message-bubble">
              <div className="loading-dots">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <form className="input-area" onSubmit={handleSubmit}>
        <input 
          type="text" 
          placeholder="Ask a question about a part..." 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
        />
        <button type="submit" className="send-button" disabled={!input.trim() || isLoading}>
          <Send size={18} />
        </button>
      </form>
    </div>
  );
}

export default App;
