"use client";

import React, { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import {
  Box,
  Heading,
  VStack,
  Input,
  Button,
  Text,
  Textarea,
} from "@chakra-ui/react";

// Simulated data (replace with actual data fetching)
const getKnowledgeBase = (id) => {
  const kbs = [
    { id: 1, name: "Keras Core API", endpointCount: 5 },
    { id: 2, name: "Keras Layers", endpointCount: 10 },
    { id: 3, name: "Keras Models", endpointCount: 3 },
  ];
  return kbs.find((kb) => kb.id === parseInt(id));
};

export default function QueryPage() {
  const params = useParams();
  const { id } = params;
  const [kb, setKb] = useState(null);
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");

  useEffect(() => {
    const knowledgeBase = getKnowledgeBase(id);
    setKb(knowledgeBase);
  }, [id]);

  const handleQuery = () => {
    // Simulate API call (replace with actual API call)
    setResponse(`Simulated response for "${query}" from ${kb.name}`);
  };

  if (!kb) {
    return <Text>Loading...</Text>;
  }

  return (
    <Box maxWidth="container.lg" margin="auto" py={10}>
      <VStack spacing={6} align="stretch">
        <Heading as="h1" size="xl" color="blue.500">
          Query {kb.name}
        </Heading>
        <Input
          placeholder="Enter your query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <Button colorScheme="blue" onClick={handleQuery}>
          Submit Query
        </Button>
        {response && (
          <Box borderWidth={1} borderRadius="md" p={4}>
            <Heading size="md" mb={2}>
              Response:
            </Heading>
            <Textarea value={response} isReadOnly rows={10} />
          </Box>
        )}
      </VStack>
    </Box>
  );
}
