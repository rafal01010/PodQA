import React, { useState, useEffect, useRef } from 'react';
import './App.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface Source {
  title: string;
  url: string;
  text: string;
}

interface ChatMessage {
  message: Message;
  sources?: Source[];
}

interface ChatSession {
  sessionId: string;
  messages: ChatMessage[];
}

// Helper function to properly convert YouTube URL to embedded format
const getYouTubeEmbedUrl = (url: string): string | null => {
  try {
    // Check if it's a YouTube URL
    if (url.includes('youtube.com') || url.includes('youtu.be')) {
      let embedUrl = '';
      
      // Parse URL to get all parameters
      const urlObj = new URL(url);
      const searchParams = new URLSearchParams(urlObj.search);
      
      // For youtube.com/watch URLs
      if (url.includes('youtube.com/watch')) {
        // Get video ID
        const videoId = searchParams.get('v');
        if (!videoId) return null;
        
        // Start with base embed URL
        embedUrl = `https://www.youtube.com/embed/${videoId}`;
        
        // Create new search params for the embed URL
        const embedParams = new URLSearchParams();
        
        // Copy important parameters (especially timestamp)
        if (searchParams.has('t')) embedParams.append('start', searchParams.get('t')!.replace('s', ''));
        if (searchParams.has('list')) embedParams.append('list', searchParams.get('list')!);
        
        // Add parameters to embed URL if any exist
        const paramString = embedParams.toString();
        if (paramString) {
          embedUrl += '?' + paramString;
        }
      }
      // For youtu.be short URLs
      else if (url.includes('youtu.be/')) {
        // Extract video ID from path
        const pathParts = urlObj.pathname.split('/');
        const videoId = pathParts[pathParts.length - 1];
        
        // Start with base embed URL
        embedUrl = `https://www.youtube.com/embed/${videoId}`;
        
        // Create new search params for the embed URL
        const embedParams = new URLSearchParams();
        
        // Copy important parameters (especially timestamp)
        if (searchParams.has('t')) embedParams.append('start', searchParams.get('t')!.replace('s', ''));
        
        // Add parameters to embed URL if any exist
        const paramString = embedParams.toString();
        if (paramString) {
          embedUrl += '?' + paramString;
        }
      }
      
      return embedUrl || null;
    }
    return null;
  } catch (error) {
    console.error("Error parsing YouTube URL:", error);
    return null;
  }
};

// YouTube Embed Component
const YouTubeEmbed: React.FC<{ embedUrl: string, onClose: () => void }> = ({ embedUrl, onClose }) => {
  // Append autoplay parameter to the URL
  const autoplayUrl = embedUrl.includes('?') ? 
    `${embedUrl}&autoplay=1` : 
    `${embedUrl}?autoplay=1`;
  
  return (
    <div className="youtube-embed-container">
      <div className="youtube-embed-header">
        <h3>YouTube Video</h3>
        <button className="close-embed-btn" onClick={onClose}>×</button>
      </div>
      <iframe 
        width="100%" 
        height="315" 
        src={autoplayUrl}
        title="YouTube video player" 
        frameBorder="0" 
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
        allowFullScreen>
      </iframe>
    </div>
  );
};


function App() {
  const [input, setInput] = useState<string>('');
  const [session, setSession] = useState<ChatSession>({
    sessionId: '',
    messages: []
  });
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [expandedSource, setExpandedSource] = useState<string | null>(null);
  const [openDropdowns, setOpenDropdowns] = useState<{[key: string]: boolean}>({});
  // New state for tracking embedded videos
  const [embeddedVideo, setEmbeddedVideo] = useState<{messageIndex: number, sourceTitle: string, sourceIndex: number, embedUrl: string} | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Initialize session when component mounts
  useEffect(() => {
    const storedSessionId = localStorage.getItem('sessionId');
    if (storedSessionId) {
      setSession(prev => ({ ...prev, sessionId: storedSessionId }));
      
      // Load chat history
      fetch(`http://localhost:8000/sessions/${storedSessionId}/history`)
        .then(res => {
          if (res.ok) return res.json();
          throw new Error('Failed to fetch history');
        })
        .then(data => {
          // Convert simple message history to our ChatMessage format
          const chatMessages = data.history.map((msg: Message) => ({
            message: msg,
            sources: [] // We don't have sources from history API
          }));
          
          setSession(prev => ({ ...prev, messages: chatMessages }));
        })
        .catch(err => {
          console.error('Error loading chat history:', err);
          // If session not found, create a new one
          localStorage.removeItem('sessionId');
        });
    }
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [session.messages]);

  useEffect(() => {
    // Only focus when loading changes from true to false (model finished responding)
    if (isLoading === false) {
      // Small timeout to ensure DOM updates are complete
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    }
  }, [isLoading]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message to UI immediately
    const userMessage: Message = { role: 'user', content: input };
    setSession(prev => ({
      ...prev,
      messages: [...prev.messages, { message: userMessage }]
    }));
    setIsLoading(true);
    setInput('');

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: session.sessionId,
          message: input,
          temperature: 0.3,
          max_new_tokens: 1000,
          top_k: 20,
          use_query_rewriting: true
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      
      // Save session ID if it's new
      if (!session.sessionId) {
        setSession(prev => ({ ...prev, sessionId: data.session_id }));
        localStorage.setItem('sessionId', data.session_id);
      }

      // Add assistant response with sources to UI
      setSession(prev => ({
        ...prev,
        messages: [...prev.messages, {
          message: data.message,
          sources: data.sources
        }]
      }));
    } catch (error) {
      console.error('Error:', error);
      // Add error message
      setSession(prev => ({
        ...prev,
        messages: [...prev.messages, {
          message: { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' },
          sources: []
        }]
      }));
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const startNewChat = () => {
    localStorage.removeItem('sessionId');
    setSession({
      sessionId: '',
      messages: []
    });
    setEmbeddedVideo(null);
    setOpenDropdowns({});
    setExpandedSource(null);
  };

  // Modified to embed video with original URL
  const handleVideoSource = (url: string, messageIndex: number, sourceTitle: string, sourceIndex: number) => {
    const embedUrl = getYouTubeEmbedUrl(url);
    if (embedUrl) {
      // If a video is already embedded with the same details, close it
      if (embeddedVideo && 
          embeddedVideo.messageIndex === messageIndex && 
          embeddedVideo.sourceTitle === sourceTitle &&
          embeddedVideo.sourceIndex === sourceIndex) {
        setEmbeddedVideo(null);
      } else {
        // Otherwise, embed the new video
        setEmbeddedVideo({
          messageIndex,
          sourceTitle,
          sourceIndex,
          embedUrl
        });
      }
    } else {
      // If not a YouTube URL, open in new tab as fallback
      window.open(url, '_blank');
    }
  };

  const closeEmbeddedVideo = () => {
    setEmbeddedVideo(null);
  };

  const toggleSourceDropdown = (title: string) => {
    setOpenDropdowns(prev => ({
      ...prev,
      [title]: !prev[title]
    }));
  };

  const getTextPreview = (text: string, maxLength: number = 100) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  const toggleFullText = (sourceKey: string) => {
    if (expandedSource === sourceKey) {
      setExpandedSource(null);
    } else {
      setExpandedSource(sourceKey);
    }
  };

  // Group sources by title
  const groupSourcesByTitle = (sources: Source[] | undefined) => {
    if (!sources || sources.length === 0) return {};
    
    const grouped: {[title: string]: Source[]} = {};
    
    sources.forEach(source => {
      if (!grouped[source.title]) {
        grouped[source.title] = [];
      }
      grouped[source.title].push(source);
    });
    
    return grouped;
  };

  return (
    <div className="app">
      <div className="sidebar">
        <h1>PodQA</h1>
        <button className="new-chat-btn" onClick={startNewChat}>New Chat</button>
      </div>
      
      <div className="chat-container">
        <div className="messages">
          {session.messages.length === 0 ? (
            <div className="welcome-message">
              <h2>Welcome to PodQA</h2>
              <p>Ask me anything about the Trash Taste podcast.</p>
            </div>
          ) : (
            session.messages.map((chatMsg, index) => (
              <div key={index} className={`message ${chatMsg.message.role}`}>
                <div className="message-content">
                  {chatMsg.message.content.split('\n').map((line, i) => (
                    <p key={i}>{line}</p>
                  ))}
                  
                  {/* Display sources for assistant messages */}
                  {chatMsg.message.role === 'assistant' && chatMsg.sources && chatMsg.sources.length > 0 && (
                    <div className="sources-container">
                      <div className="sources-heading">Sources:</div>
                      <div className="sources-list">
                        {Object.entries(groupSourcesByTitle(chatMsg.sources)).map(([title, sources]) => (
                          <div key={title} className="source-group">
                            {sources.length === 1 ? (
                              <div className="source-item single">
                                <div 
                                  className="source-title"
                                  onClick={() => handleVideoSource(sources[0].url, index, title, 0)}
                                >
                                  {title}
                                </div>
                                <div className="source-actions">
                                  <button 
                                    className="video-btn"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleVideoSource(sources[0].url, index, title, 0);
                                    }}
                                  >
                                    {embeddedVideo && 
                                     embeddedVideo.messageIndex === index && 
                                     embeddedVideo.sourceTitle === title &&
                                     embeddedVideo.sourceIndex === 0 ? 'Hide Video' : 'Watch Video'}
                                  </button>
                                  <button 
                                    className="view-text-btn"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      toggleFullText(`${index}-${title}-0`);
                                    }}
                                  >
                                    {expandedSource === `${index}-${title}-0` ? 'Hide' : 'View Text'}
                                  </button>
                                </div>
                                {expandedSource === `${index}-${title}-0` && (
                                  <div className="source-full-text">
                                    {sources[0].text}
                                  </div>
                                )}
                                {embeddedVideo && 
                                 embeddedVideo.messageIndex === index && 
                                 embeddedVideo.sourceTitle === title &&
                                 embeddedVideo.sourceIndex === 0 && (
                                  <YouTubeEmbed 
                                    embedUrl={embeddedVideo.embedUrl} 
                                    onClose={closeEmbeddedVideo} 
                                  />
                                )}
                              </div>
                            ) : (
                              <div className="source-item multiple">
                                <div 
                                  className="source-title with-dropdown" 
                                  onClick={() => toggleSourceDropdown(title)}
                                >
                                  {title} ({sources.length})
                                  <span className={`dropdown-arrow ${openDropdowns[title] ? 'open' : ''}`}>▼</span>
                                </div>
                                {openDropdowns[title] && (
                                  <div className="source-dropdown">
                                    {sources.map((source, i) => (
                                      <div key={i} className="dropdown-item">
                                        <div className="source-preview">
                                          {getTextPreview(source.text)}
                                        </div>
                                        <div className="dropdown-actions">
                                          <button
                                            className="video-btn"
                                            onClick={() => handleVideoSource(source.url, index, title, i)}
                                          >
                                            {embeddedVideo && 
                                             embeddedVideo.messageIndex === index && 
                                             embeddedVideo.sourceTitle === title &&
                                             embeddedVideo.sourceIndex === i ? 'Hide Video' : 'Watch Video'}
                                          </button>
                                          <button 
                                            className="view-text-btn"
                                            onClick={() => toggleFullText(`${index}-${title}-${i}`)}
                                          >
                                            {expandedSource === `${index}-${title}-${i}` ? 'Hide' : 'View Text'}
                                          </button>
                                        </div>
                                        {expandedSource === `${index}-${title}-${i}` && (
                                          <div className="source-full-text">
                                            {source.text}
                                          </div>
                                        )}
                                        {embeddedVideo && 
                                         embeddedVideo.messageIndex === index && 
                                         embeddedVideo.sourceTitle === title &&
                                         embeddedVideo.sourceIndex === i && (
                                          <YouTubeEmbed 
                                            embedUrl={embeddedVideo.embedUrl} 
                                            onClose={closeEmbeddedVideo} 
                                          />
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="message assistant">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <form className="input-form" onSubmit={handleSendMessage}>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={isLoading}
        />
          <button type="submit" disabled={isLoading || !input.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;