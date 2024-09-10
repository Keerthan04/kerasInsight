'use client'
import React, { useState } from "react";
import {
  Box,
  Button,
  VStack,
  Heading,
  Text,
  Textarea,
  Input,
  Progress,
  useToast,
} from "@chakra-ui/react";
import { ChevronRight, ChevronLeft } from "lucide-react";

const steps = [
  {
    title: "Input API Endpoints",
    description: "Enter the Keras API endpoints you want to explore.",
  },
  {
    title: "Create Knowledge Base",
    description:
      "We'll process the API documentation to create a knowledge base.",
  },
  {
    title: "Ask Questions",
    description: "Ask questions about the Keras API and get instant answers.",
  },
];

export default function GetStartedForm() {
  const [currentStep, setCurrentStep] = useState(0);
  const [apiEndpoints, setApiEndpoints] = useState("");
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();

  const handleNext = async () => {
    if (currentStep === 0 && !apiEndpoints.trim()) {
      toast({
        title: "Error",
        description: "Please enter at least one API endpoint.",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    if (currentStep === 1) {
      setIsLoading(true);
      try {
        // Simulating API call to create knowledge base
        await new Promise((resolve) => setTimeout(resolve, 2000));
        toast({
          title: "Success",
          description: "Knowledge base created successfully.",
          status: "success",
          duration: 3000,
          isClosable: true,
        });
      } catch (error) {
        toast({
          title: "Error",
          description: "Failed to create knowledge base. Please try again.",
          status: "error",
          duration: 3000,
          isClosable: true,
        });
        return;
      } finally {
        setIsLoading(false);
      }
    }

    setCurrentStep((prev) => prev + 1);
  };

  const handleBack = () => {
    setCurrentStep((prev) => prev - 1);
  };

  const handleSubmitQuery = async () => {
    if (!query.trim()) {
      toast({
        title: "Error",
        description: "Please enter a query.",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setIsLoading(true);
    try {
      // Simulating API call to get response
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setResponse(
        "This is a simulated response to your query about the Keras API. In a real implementation, this would be the actual response from your backend."
      );
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to get response. Please try again.",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box maxWidth="container.md" margin="auto" py={10}>
      <Progress
        value={(currentStep / (steps.length - 1)) * 100}
        mb={8}
        colorScheme="red"
      />
      <VStack spacing={8} align="stretch">
        <Heading as="h2" size="xl" color="red.500">
          {steps[currentStep].title}
        </Heading>
        <Text>{steps[currentStep].description}</Text>

        {currentStep === 0 && (
          <Textarea
            value={apiEndpoints}
            onChange={(e) => setApiEndpoints(e.target.value)}
            placeholder="Enter API endpoints (one per line)"
            rows={6}
          />
        )}

        {currentStep === 2 && (
          <>
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your question about the Keras API"
            />
            <Button
              onClick={handleSubmitQuery}
              colorScheme="red"
              isLoading={isLoading}
            >
              Submit Query
            </Button>
            {response && (
              <Box borderWidth={1} borderRadius="md" p={4} mt={4}>
                <Text fontWeight="bold">Response:</Text>
                <Text>{response}</Text>
              </Box>
            )}
          </>
        )}

        <Box>
          {currentStep > 0 && (
            <Button onClick={handleBack} leftIcon={<ChevronLeft />} mr={4}>
              Back
            </Button>
          )}
          {currentStep < steps.length - 1 && (
            <Button
              onClick={handleNext}
              rightIcon={<ChevronRight />}
              colorScheme="red"
              isLoading={isLoading}
            >
              {currentStep === 1 ? "Create Knowledge Base" : "Next"}
            </Button>
          )}
        </Box>
      </VStack>
    </Box>
  );
}
