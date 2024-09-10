"use client";

import React, { useState } from "react";
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  VStack,
  HStack,
  Input,
  Button,
  Text,
  Box,
} from "@chakra-ui/react";

const ChatWindow = ({ knowledgeBase, isOpen, onClose }) => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");

  const handleSendMessage = () => {
    if (inputMessage.trim() === "") return;

    const newMessage = {
      text: inputMessage,
      sender: "user",
    };

    setMessages([...messages, newMessage]);
    setInputMessage("");

    // Simulate a response (replace this with actual API call)
    setTimeout(() => {
      const botResponse = {
        text: `Response from ${knowledgeBase.name}: ${inputMessage}`,
        sender: "bot",
      };
      setMessages((prevMessages) => [...prevMessages, botResponse]);
    }, 1000);
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Chat with {knowledgeBase.name}</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <VStack spacing={4} align="stretch" height="400px">
            <Box
              flex={1}
              overflowY="auto"
              borderWidth={1}
              borderRadius="md"
              p={2}
            >
              {messages.map((message, index) => (
                <Text
                  key={index}
                  textAlign={message.sender === "user" ? "right" : "left"}
                >
                  <strong>{message.sender === "user" ? "You" : "Bot"}:</strong>{" "}
                  {message.text}
                </Text>
              ))}
            </Box>
            <HStack>
              <Input
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Type your message..."
                onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
              />
              <Button onClick={handleSendMessage} colorScheme="blue">
                Send
              </Button>
            </HStack>
          </VStack>
        </ModalBody>
        <ModalFooter>
          <Button onClick={onClose}>Close</Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default ChatWindow;
